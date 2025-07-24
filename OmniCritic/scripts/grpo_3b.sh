#!/usr/bin/env bash
# ---------- 环境变量 ----------
export MAX_PIXELS=262144
export VIDEO_MAX_PIXELS=65536
export NPROC_PER_NODE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FPS=1
export FPS_MAX_FRAMES=20
export AUDIO_MEMMAP=true
export AUDIO_CHUNK_SIZE=4000
export AUDIO_NUM_WORKERS=4

# ---------- GRPO 训练 ----------
swift rlhf \
  --rlhf_type grpo \
  --model /data/yiwei.ru/modelscope/Qwen/Qwen2.5-Omni-3B \
  --reward_funcs mm_format mm_content mm_rubric \
  --reward_weights 0.4 0.4 0.2 \
  --train_type lora \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --torch_dtype bfloat16 \
  --dataset \
      /data/yiwei.ru/omniR1-sft-master/OmniCritic/sft_dataset/audio/final_final_rl_data.jsonl \
      /data/yiwei.ru/omniR1-sft-master/OmniCritic/sft_dataset/video/final_final_rl_data.jsonl \
      /data/yiwei.ru/omniR1-sft-master/OmniCritic/sft_dataset/image/final_final_rl_data.jsonl \
  --external_plugins /data/yiwei.ru/ms-swift-main/examples/train/grpo/plugin/new_plugin.py \
  --max_completion_length 2048 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --eval_steps 100 \
  --save_steps 100 \
  --save_total_limit 2 \
  --logging_steps 5 \
  --max_length 8192 \
  --output_dir /data/yiwei.ru/omniR1-sft-master/OmniCritic/outputs/grpo \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --dataset_num_proc 4 \
  --num_generations 4 \
  --temperature 1.0 \
  --top_p 0.99 \
  --top_k 50 \
  --lazy_tokenize false \
  --system /data/yiwei.ru/omniR1-sft-master/OmniCritic/utils/prompt.txt \
  --deepspeed zero2 \
  --log_completions true \
  2>&1 | tee /data/yiwei.ru/omniR1-sft-master/OmniCritic/logs/log_file_grpo_3b.txt
