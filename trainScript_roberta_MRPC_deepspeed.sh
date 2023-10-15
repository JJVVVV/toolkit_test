#!/bin/bash
# ./trainScript_roberta_MRPC_deepspeed.sh
# nohup ./trainScript_roberta_MRPC_deepspeed.sh > /dev/null 2>&1 &
# pkill -s SIGKILL -pgn 3697090

# 定义一个数组，存放种子
# while kill -0 $PID 2>/dev/null; do sleep 1; done


# # QQP
# 25571765677776765
seed=17
CUDA_VISIBLE_DEVICES=0,1

# ###################################parameters#########################################
# task_type="classify"
task_type="generate"

dashboard="None"
dataset_name="MRPC"
part="all"
text_type='ORI'

min_threshold=None
alpha=0.2

# model_type="bert-base-uncased"
if [ "$task_type" = "classify" ]; then
  model_type="roberta-base"
else
  model_type="google/flan-t5-base"
fi
model_dir="/data/jjwang/pretrained/$model_type"
# model_type="bert-large-uncased"
# model_type='hfl/chinese-bert-wwm-ext'

model_name="deepspeed_fp16_baseline"
# model_name="noise2_$min_threshold"
# model_name="deepspeed_fp16_shift1_$alpha"
# model_name="rotate"
# model_name="rotate_only"
# model_name="noise_only_$min_threshold"
if [ "$task_type" = "classify" ]; then
  fp16=True
  bf16=False
fi
if [ "$task_type" = "generate" ]; then
  fp16=False
  bf16=True
fi

test_in_epoch=False

accumulate_step=10
train_batch_size=500
infer_batch_size=500
epochs=3
max_length_input=None
learning_rate='2e-4'
warmup_num_step=-1
warmup_ratio_step=0.1
if [ "$task_type" = "classify" ]; then
  metric='Accuracy'
else
  metric='rougeL'
fi

save_dir="outputs"

max_new_tokens=10
do_sample=True
num_beams=1
top_k=5
temperature=0.01


train_file_path="data/$dataset_name/train/$part.jsonl"
val_file_path="data/$dataset_name/validation/$part.jsonl"
# test_file_path="data/$dataset_name/test/$part.jsonl"
test_file_path=None

warmup_ratio=0.1
# ###################################parameters#########################################
# deepspeed --include localhost \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
torchrun \
  --rdzv-backend=c10d \
  --rdzv-endpoint=localhost:0 \
  --nnodes=1 \
  --nproc-per-node=2 \
  ./train_trainer.py \
    --task_type $task_type \
    --dataset_name $dataset_name \
    --model_type $model_type \
    --model_name $model_name \
    --train_file_path $train_file_path \
    --val_file_path $val_file_path \
    --test_file_path $test_file_path \
    --seed $seed \
    --opt_lr $learning_rate \
    --epochs $epochs \
    --train_batch_size $train_batch_size \
    --infer_batch_size $infer_batch_size \
    --sch_warmup_num_steps $warmup_num_step \
    --sch_warmup_ratio_steps $warmup_ratio_step \
    --max_length_input $max_length_input \
    --metric $metric \
    --eval_every_half_epoch $test_in_epoch \
    --gradient_accumulation_steps $accumulate_step \
    --opt_weight_decay $weight_decay \
    --dashboard $dashboard \
    --text_type $text_type \
    --min_threshold $min_threshold \
    --alpha $alpha \
    --part $part \
    --model_dir $model_dir \
    --parallel_mode deepspeed \
    --fp16 $fp16 \
    --bf16 $bf16 \
    --deepspeed_config ds_zero3_offload.hjson \
    --max_new_tokens $max_new_tokens \
    --do_sample $do_sample \
    --num_beams $num_beams \
    --top_k $top_k \
    --temperature $temperature \
    --logging_steps 1 \
    --save_dir $save_dir \
    --save_latest_ckpt False \
    --test_load_to_gpu_directly True \
    --padding_side right \
    --ddp_timeout 3000

    # > $log_file 2>&1 &
cd "$save_dir/optimal_checkpoint" && python zero_to_fp32.py . pytorch_model.bin && rm -rf optimal
