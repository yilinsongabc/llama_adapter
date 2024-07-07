export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

python -u -m torch.distributed.launch --master_port=1112 --nproc_per_node=7 --use_env \
  main_finetune.py --data_config "/barn6/yilinsong/tmp/dataset1.json" --batch_size 4 \
 --epochs 4 --warmup_epochs 1 --blr 10e-4 --weight_decay 0.02 \
 --llama_path "/barn6/yilinsong/LLaMA-Adapter/checkpoints" \
 --output_dir "/barn6/yilinsong/LLaMA-Adapter/llama_adapter_v2_multimodal7b/exps/test_finetune_query100" \
 --pretrained_path "/barn6/yilinsong/LLaMA-Adapter/checkpoints/llama_adapter_len10_layer30_release.pth" \
 --num_workers 4
# &>> "/barn6/yilinsong/LLaMA-Adapter/llama_adapter_v2_multimodal7b/exps/test"/output.log &
# /barn6/yilinsong/llava/LLaVA/playground/data/llava_v1_5_mix665k.json
# /barn6/yilinsong/tmp/dataset1.json
