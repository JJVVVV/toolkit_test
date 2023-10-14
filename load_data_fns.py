from enum import Enum, auto
from pathlib import Path

import jsonlines
import numpy as np
import pandas as pd
from toolkit.enums import Split
from toolkit.nlp.data import ClassificationLabel, PairedText
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class DatasetName(Enum):
    QQP = auto()
    SST2 = auto()
    MNLI = auto()
    QNLI = auto()
    MRPC = auto()
    RTE = auto()


class TextType(Enum):
    ANS = auto()
    DIS = auto()
    QUE_DIS = auto()
    QUE_ANS = auto()
    QUE_PSEUDO = auto()

    ORI = auto()
    DATA_AUG_REP2 = auto()
    DATA_AUG_REP4 = auto()
    DATA_AUG_REP4_FUSED = auto()
    DATA_AUG_REP4_CLOSED = auto()
    DATA_AUG_REP6 = auto()
    GAUSSIAN_LABEL = auto()
    SORTED_DATA = auto()


DATASET_CLASSNUM_MAP = {DatasetName.QQP: 2, DatasetName.MRPC: 2, DatasetName.MNLI: 3, DatasetName.QNLI: 2, DatasetName.SST2: 2, DatasetName.RTE: 2}


def get_sep_token_num(model_type):
    if "roberta" in model_type:
        return 2
    else:
        return 1


def load_data_fn_mrpc_rte(
    data_file_path: Path | str, model_type: str, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, split: Split, **kwargs
):
    special_tokens_map = tokenizer.special_tokens_map
    BOS = special_tokens_map["bos_token"] if "bos_token" in special_tokens_map.keys() else None
    EOS = special_tokens_map["eos_token"] if "eos_token" in special_tokens_map.keys() else None
    SEP = special_tokens_map["sep_token"] if "sep_token" in special_tokens_map.keys() else None
    MASK = special_tokens_map["mask_token"] if "mask_token" in special_tokens_map.keys() else None
    text_type = kwargs["text_type"]
    sep_num = get_sep_token_num(model_type)

    with jsonlines.open(data_file_path, "r") as jlReader:
        dict_objs = list(jlReader)
        if isinstance(dict_objs[0], str):
            dict_objs = dict_objs[1:]

    dict_objs = dict_objs[:1000]

    inputs = []
    labels = []
    for dict_obj in dict_objs:
        if text_type == TextType.ORI:
            inputs.append(PairedText(dict_obj["sentence1"], dict_obj["sentence2"]))
        labels.append(ClassificationLabel(dict_obj["label"]))
    return inputs, labels


def load_data_fn_mrpc_rte_gen(
    data_file_path: Path | str, model_type: str, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, split: Split, **kwargs
):
    special_tokens_map = tokenizer.special_tokens_map
    BOS = special_tokens_map["bos_token"] if "bos_token" in special_tokens_map.keys() else None
    EOS = special_tokens_map["eos_token"] if "eos_token" in special_tokens_map.keys() else None
    SEP = special_tokens_map["sep_token"] if "sep_token" in special_tokens_map.keys() else None
    MASK = special_tokens_map["mask_token"] if "mask_token" in special_tokens_map.keys() else None
    text_type = kwargs["text_type"]
    sep_num = get_sep_token_num(model_type)

    with jsonlines.open(data_file_path, "r") as jlReader:
        dict_objs = list(jlReader)
        if isinstance(dict_objs[0], str):
            dict_objs = dict_objs[1:]

    dict_objs = dict_objs[:1000]
    inputs = []
    labels = []
    for dict_obj in dict_objs:
        if text_type == TextType.ORI:
            inputs.append(PairedText("Are the following two sentences the same meaning?\n" + dict_obj["sentence1"] + " " + dict_obj["sentence2"]))
        text_label = "Yes, they have the same meaning." if dict_obj["label"] == 1 else "No, they have different meanings."
        if split == Split.TRAINING:
            labels.append(PairedText(text_label))
        else:
            labels.append(text_label)
    return inputs, labels


def load_data_fn_mnli(data_file_path: Path | str, model_type: str, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, split: Split, **kwargs):
    special_tokens_map = tokenizer.special_tokens_map
    BOS = special_tokens_map["bos_token"] if "bos_token" in special_tokens_map.keys() else None
    EOS = special_tokens_map["eos_token"] if "eos_token" in special_tokens_map.keys() else None
    SEP = special_tokens_map["sep_token"] if "sep_token" in special_tokens_map.keys() else None
    MASK = special_tokens_map["mask_token"] if "mask_token" in special_tokens_map.keys() else None
    text_type = kwargs["text_type"]
    sep_num = get_sep_token_num(model_type)

    with jsonlines.open(data_file_path, "r") as jlReader:
        dict_objs = list(jlReader)
        if isinstance(dict_objs[0], str):
            dict_objs = dict_objs[1:]

    inputs = []
    labels = []
    for dict_obj in dict_objs:
        if text_type == TextType.ORI:
            inputs.append(PairedText(dict_obj["premise"], dict_obj["hypothesis"]))
        labels.append([dict_obj["label"]])
    return inputs, labels


def load_data_fn_qnli(data_file_path: Path | str, model_type: str, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, split: Split, **kwargs):
    pass


def load_data_fn_qqp():
    pass


def load_data_fn_sst2(data_file_path: Path | str, model_type: str, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, split: Split, **kwargs):
    pass


LOAD_DATA_FNS = {
    "classify": {
        DatasetName.QQP: load_data_fn_qqp,
        DatasetName.SST2: load_data_fn_sst2,
        DatasetName.MNLI: load_data_fn_mnli,
        DatasetName.QNLI: load_data_fn_qnli,
        DatasetName.MRPC: load_data_fn_mrpc_rte,
        DatasetName.RTE: load_data_fn_mrpc_rte,
    },
    "generate": {DatasetName.MRPC: load_data_fn_mrpc_rte_gen},
}
