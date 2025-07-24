export MAX_PIXELS=262144
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FPS=1
export FPS_MAX_FRAMES=20
export VIDEO_MAX_PIXELS=32768
export AUDIO_MEMMAP=true
export AUDIO_CHUNK_SIZE=4000
export AUDIO_NUM_WORKERS=4
export VIDEO_READER_BACKEND=torchvision
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export SWIFT_DEVICE_MAP=auto

torchrun --nproc_per_node=4 /data/yiwei.ru/omniR1-sft-master/ms-swift-main/swift/cli/sft.py \
  --do_train \
  --model /data/yiwei.ru/modelscope/Qwen/Qwen2.5-Omni-3B \
  --dataset /data/yiwei.ru/omniR1-sft-master/OmniCritic/sft_dataset/audio/final_sft_data.jsonl \
            /data/yiwei.ru/omniR1-sft-master/OmniCritic/sft_dataset/video/final_sft_data.jsonl \
            /data/yiwei.ru/omniR1-sft-master/OmniCritic/sft_dataset/image/final_sft_data.jsonl \
  --lora_rank 8 \
  --lora_alpha 32 \
  --freeze_vit true \
  --target_modules all-linear \
  --output_dir /data/yiwei.ru/omniR1-sft-master/OmniCritic/outputs \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 12 \
  --learning_rate 5e-6 \
  --warmup_ratio 0.03 \
  --max_length 3072 \
  --eval_steps 100 \
  --save_steps 100 \
  --save_total_limit 4 \
  --logging_steps 5 \
  --dataloader_num_workers 2 \
  --torch_dtype bfloat16 \
  --deepspeed zero2 \
  2>&1 | tee /data/yiwei.ru/omniR1-sft-master/OmniCritic/logs/log_file_3b_lora.txt
