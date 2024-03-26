#!/usr/bin/bash

# LLAMA_PATH="$1"
# CONFIG="$2"
# OUTPUT_DIR="$3"

# mkdir -p "$OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

python -u -m torch.distributed.launch --master_port=1112 --nproc_per_node=7 --use_env \
 main_pretrain.py --data_config "/barn6/yilinsong/tmp/dataset1.json" --batch_size 4 \
 --epochs 150 --split_epoch 50 --warmup_epochs 5 --blr 1.0e-4 --weight_decay 0.05 \
 --llama_path "/barn6/yilinsong/LLaMA-Adapter/checkpoints" \
 --output_dir "/barn6/yilinsong/LLaMA-Adapter/llama_adapter_v2_multimodal7b/exps/test_pretrain" \
#  &>> "$OUTPUT_DIR"/output.log &