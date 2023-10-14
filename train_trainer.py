import logging
import time
from pathlib import Path

import deepspeed
import hjson
import numpy as np
import toolkit
import torch
import torch.distributed as dist
import wandb
from fire import Fire
from sklearn.metrics import accuracy_score, f1_score
from toolkit import getLogger
from toolkit.enums import Split
from toolkit.metric import MetricDict, calculate_rouge
from toolkit.nlp import NLPTrainingConfig, TextDataset
from toolkit.training import CheckpointManager, Trainer
from toolkit.training.initializer import initialize
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from load_data_fns import DATASET_CLASSNUM_MAP, LOAD_DATA_FNS, DatasetName, TextType
from model.MatchModel_binary_classification import RobertaModel_binary_classify
from model.MatchModel_generation import T5Model

deepspeed.logger.setLevel(logging.INFO)


def load_dataset(tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast) -> tuple:
    # * Load training data, development data and test data
    train_dataset = TextDataset.from_file(
        configs.train_file_path,
        tokenizer,
        split=Split.TRAINING,
        configs=configs,
        load_data_fn=LOAD_DATA_FNS[configs.task_type][DATASETNAME],
        text_type=TEXTTYPE,
    )
    if dist.is_initialized():
        dist.barrier()
    val_dataset = TextDataset.from_file(
        configs.val_file_path,
        tokenizer,
        split=Split.VALIDATION,
        configs=configs,
        load_data_fn=LOAD_DATA_FNS[configs.task_type][DATASETNAME],
        text_type=TEXTTYPE,
    )
    if dist.is_initialized():
        dist.barrier()
    test_dataset = TextDataset.from_file(
        configs.test_file_path,
        tokenizer,
        split=Split.TEST,
        configs=configs,
        load_data_fn=LOAD_DATA_FNS[configs.task_type][DATASETNAME],
        text_type=TEXTTYPE,
    )
    if dist.is_initialized():
        dist.barrier()
    return train_dataset, val_dataset, test_dataset


def calculate_metric_callback(all_labels: list, all_logits: list, mean_loss: float) -> MetricDict:
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    match DATASET_CLASSNUM_MAP[DATASETNAME]:
        case 2:
            if "DATA_AUG_REP" in TEXTTYPE.name and "FUSED" not in TEXTTYPE.name:
                # all_logits: (num, 4)
                all_ori_preds = (all_logits > 0).astype(int)
                threshold = all_ori_preds.shape[1] >> 1
                vote_pos = all_ori_preds.sum(axis=1)
                all_preds = np.zeros_like(vote_pos)
                pos_mask = vote_pos > threshold
                neg_mast = ~pos_mask
                controversial_mask = np.zeros_like(pos_mask).astype(bool) if all_ori_preds.shape[1] & 1 else vote_pos == threshold
                all_preds[pos_mask] = 1
                all_preds[neg_mast] = 0
                all_preds[controversial_mask] = all_ori_preds[controversial_mask][:, 0]
                # definite_mask = (vote_pos == all_ori_preds.shape[1]) | (vote_pos == 0)
                # confused_mask = ~(definite_mask | controversial_mask)
            elif TEXTTYPE == TextType.GAUSSIAN_LABEL:
                # all_logtis: (num, 100)
                all_preds = (np.argmax(all_logits, axis=1, keepdims=True) >= 50).astype(int)
            else:
                # all_logtis: (num, 1)
                all_preds = (all_logits > 0).astype(int)
        case _:
            all_preds = np.argmax(all_logits, axis=1, keepdims=True)

    if DATASET_CLASSNUM_MAP[DATASETNAME] == 2:
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="binary")
    else:
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="micro")

    return MetricDict({"Accuracy": acc * 100, "F1-score": f1 * 100, "Loss": mean_loss})


def calculate_metric_callback_gen(all_labels: list, all_logits: list, mean_loss: float) -> MetricDict:
    return calculate_rouge(all_labels, all_logits, ("rougeL", "rouge1", "rouge2"), "en")


def load_model() -> tuple[PreTrainedModel | DDP, PreTrainedTokenizer | PreTrainedTokenizerFast, int]:
    # * Determine the model architecture
    global DATASETNAME
    match DATASETNAME:
        case DatasetName.MRPC:
            if "classify" == configs.task_type:
                MatchModel = RobertaModel_binary_classify
            elif "generate" == configs.task_type:
                MatchModel = T5Model

    # * Determine the model path
    if ckpt_manager.latest_id == -1:
        pretrainedModelDir = configs.model_dir if configs.model_dir is not None else configs.model_type
    else:
        pretrainedModelDir = ckpt_manager.latest_dir
    logger.debug(f"local_rank {local_rank}: load model from {pretrainedModelDir}")

    # * Load model, tokenizer to CPU memory
    logger.debug(f"local_rank {local_rank}: Loading model and tokenizer to CPU memory...")
    start = time.time()
    # 加载自定义配置
    my_config = None
    try:
        my_config = AutoConfig.from_pretrained(f"config/my_{configs.model_type}_config")
        logger.debug(str(my_config))
    except:
        pass

    tokenizer = AutoTokenizer.from_pretrained(pretrainedModelDir)
    model = MatchModel.from_pretrained(pretrainedModelDir)
    end = time.time()

    logger.debug(f"local_rank {local_rank}: Loading model and tokenizer from disk to CPU memory takes {end - start:.2f} sec.")
    if dist.is_initialized():
        dist.barrier()
    return model, tokenizer


@record
def main() -> None:
    # * Request GPU memory
    # allocate_gpu_memory(0.8)
    # # time.sleep(99999)

    # * Loading model
    model, tokenizer = load_model()
    if configs.dashboard == "wandb":
        wandb.watch(model.module if hasattr(model, "module") else model, log_freq=256)

    # * load dataset
    train_dataset, val_dataset, test_dataset = load_dataset(tokenizer)

    # * Train
    trainer = Trainer(
        task_type=configs.task_type,
        evaluate_only=False,
        config=configs,
        model=model,
        dataset_train=train_dataset,
        dataset_val=val_dataset,
        dataset_test=test_dataset,
        calculate_metric_callback=calculate_metric_callback if configs.task_type == "classify" else calculate_metric_callback_gen,
        optimizer="AdamW",
        scheduler="LinearWarmup",
        tokenizer=tokenizer,
        dashboard_writer=run,
        extral_args_training={"is_train": True},
        extral_args_evaluation={"is_train": False},
    )
    trainer.train()
    # time.sleep(3)


if __name__ == "__main__":
    # * Get args
    configs: NLPTrainingConfig = Fire(NLPTrainingConfig, silence=True)

    # * Create checkpoints,tensorboard outputs directory
    _dir = Path(
        configs.task_type,
        configs.dataset_name,
        configs.model_type,
        configs.text_type,
        configs.part,
        configs.model_name,
        str(configs.epochs),
        str(configs.train_batch_size),
        str(configs.opt_lr),
        str(configs.seed),
    )
    if configs.save_dir is None:
        configs.save_dir = Path("outputs", _dir)
    else:
        pass
    configs.save(configs.save_dir, silence=False)

    # * Create checkpoint manager
    ckpt_manager = CheckpointManager(configs.save_dir)

    # * Create logger
    output_path_logger = configs.save_dir / "report.log"
    logger = getLogger(__name__, output_path_logger)
    toolkit.set_file_logger(output_path_logger)

    # * Initalize parallel and seed
    local_rank, world_size = initialize(configs)

    # * Global variable
    DATASETNAME = DatasetName[configs.dataset_name]
    TEXTTYPE = TextType[configs.text_type]

    # * Create tensorboard writer
    if configs.dashboard is None:
        run = None
        main()
    else:
        if local_rank == 0:
            if configs.dashboard == "wandb":
                with wandb.init(
                    # mode="disabled",
                    project="second",
                    config=configs.to_dict(),
                    group=f"{configs.dataset_name},train_data={configs.part}",
                    tags=[configs.dataset_name, configs.model_type, configs.model_name, configs.text_type],
                ) as run:
                    assert run is wandb.run
                    main()
            elif configs.dashboard == "tensorboard":
                run_dir = Path("tensorboard", _dir, "logs")
                run_dir.mkdir(parents=True, exist_ok=True)
                with SummaryWriter(comment="training", log_dir=run_dir) as run:
                    main()
        else:
            run = None
            main()

    # if configs.seed == 5:
    #     time.sleep(99999)
