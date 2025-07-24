# OmniCritic-R1



本项目包含 OmniCritic-R1 的完整代码与数据，用于复现论文中的多模态奖励模型训练与评估流程。



## 🔧 环境准备



建议使用以下环境配置：



- Python >= 3.10

- CUDA >= 11.8

- 推荐使用 Linux + 多卡 GPU



1. 创建并激活环境、下载依赖和模型：



conda create -n omnicritic python=3.10 -y

conda activate omnicritic

pip install -r requirements.txt

cd omniR1-sft-master/ms-swift-main

pip install -e .





下载qwen2.5omni3b和7b

以3b为例

from modelscope.hub.snapshot_download import snapshot_download



model_dir = snapshot_download(

    'qwen/Qwen2.5-Omni-3B',

    cache_dir='./Qwen2.5-Omni-3B',  # ✅ 可指定本地路径

    revision='master'

)





 2. 克隆本项目

在你希望存放的路径下执行：

git clone https://github.com/tmfk418/omni.git

cd omni



4. 数据准备



需要下载的数据集托管在 Hugging Face 上。可以使用 huggingface_hub 库进行下载。



安装 huggingface_hub 库：



pip install huggingface_hub



下载数据集：



from huggingface_hub import hf_hub_download



下载音频、视频和图像数据集



image_dataset = hf_hub_download("TMFK/omnir1-dataset", "rlaif-v-dataset.tar.gz")



video_dataset = hf_hub_download("TMFK/omnir1-dataset", "C:Userstmfk1videovideo.zip")



audio_dataset = hf_hub_download("TMFK/omnir1-dataset", "Clotho-AQA dataset.zip")



5.更新sft和rl的数据集路径

运行以下脚本来更新音频和视频路径：



更新路径



python configs/change_path.py



确保路径与您下载的文件相匹配。脚本会根据指定的基础路径更新 JSONL 文件中的sft和rl图像、音频和视频路径。

辛苦转换后看一下是不是和实际的数据集的路径相符



6. 运行sft代码

bash scripts/sft_3b.sh

bash scripts/sft_3b_lora.sh

bash scripts/sft_7b.sh

bash scripts/sft_7b_lora.sh



注意修改里面的所有路径

以sft_3b.sh为例



# ✅ 根据实际显卡数和资源配置修改

export CUDA_VISIBLE_DEVICES=0,1,2,3        # ← ✅ 若用单卡或不同编号需修改



# ✅ 可选项，根据任务需要设置（图像/视频/音频）

export MAX_PIXELS=262144

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



# ✅ 启动训练主命令（根据模型、路径修改）

torchrun --nproc_per_node=4                         # ← ✅ 改为你的显卡数（如 1、2、4、8）

  /data/yiwei.ru/omniR1-sft-master/ms-swift-main/swift/cli/sft.py 



  --do_train 



  # ✅ 模型路径：修改为你本地/远程的基座模型路径

  --model /data/yiwei.ru/modelscope/Qwen/Qwen2.5-Omni-3B         # ← ✅ 替换为你使用的模型路径



  # ✅ 数据集路径：使用你准备的数据集（建议绝对路径）

  --dataset 

    /data/yiwei.ru/omniR1-sft-master/OmniCritic/sft_dataset/audio/final_sft_data.jsonl 

    /data/yiwei.ru/omniR1-sft-master/OmniCritic/sft_dataset/video/final_sft_data.jsonl 

    /data/yiwei.ru/omniR1-sft-master/OmniCritic/sft_dataset/image/final_sft_data.jsonl      # ← ✅ 替换为你准备的 jsonl 文件



  --train_type full                      



  # ✅ 输出路径：训练结果会保存在这里

  --output_dir /data/yiwei.ru/omniR1-sft-master/OmniCritic/outputs    # ← ✅ 根据你想保存的位置修改



  # ✅ 训练参数（辛苦根据实际情况调整）

  --num_train_epochs 2 

  --per_device_train_batch_size 1 

  --per_device_eval_batch_size 1 

  --gradient_accumulation_steps 12 

  --learning_rate 5e-6 

  --warmup_ratio 0.03 

  --max_length 3072 

  --eval_steps 100 

  --save_steps 100 

  --save_total_limit 4 

  --logging_steps 5 

  --dataloader_num_workers 2 

  --torch_dtype bfloat16                

  --deepspeed zero2 



  # ✅ 输出日志（建议修改路径）

  2>&1 | tee /data/yiwei.ru/omniR1-sft-master/OmniCritic/logs/log_file_3b.txt   # ← ✅ 可替换为你本地的 logs 路径



7.运行grpo代码



bash scripts/grpo_3b.sh

bash scripts/grpo_7b.sh



export MAX_PIXELS=262144                     

export VIDEO_MAX_PIXELS=65536                

export NPROC_PER_NODE=4                       # ✅ 使用 GPU 数量（必须与 CUDA_VISIBLE_DEVICES 数量一致）

export CUDA_VISIBLE_DEVICES=0,1,2,3           # ✅ 设置实际可用显卡编号

export FPS=1

export FPS_MAX_FRAMES=20

export AUDIO_MEMMAP=true

export AUDIO_CHUNK_SIZE=4000

export AUDIO_NUM_WORKERS=4



swift rlhf 

  --rlhf_type grpo 



  # ✅ 模型路径（替换为你自己的预训练模型，如 3B 或 7B）

  --model /data/yiwei.ru/modelscope/Qwen/Qwen2.5-Omni-3B 

  --reward_funcs mm_format mm_content mm_rubric 

  --reward_weights 0.4 0.4 0.2 

  --train_type lora 

  --lora_rank 8 

  --lora_alpha 32 

  --target_modules all-linear 

  --torch_dtype bfloat16                     

  # ✅ 多模态 RL 数据路径（替换为你准备的 jsonl 文件）

  --dataset 

      /data/.../audio/final_final_rl_data.jsonl 

      /data/.../video/final_final_rl_data.jsonl 

      /data/.../image/final_final_rl_data.jsonl 



  # ✅ 外部插件：自定义奖励函数 / 数据解析（已经放在ms—swift-main的里面了，只改路径前缀就好）

  --external_plugins /path/to/new_plugin.py 



  # 模型生成与训练参数

  --max_completion_length 2048 

  --num_train_epochs 1 

  --per_device_train_batch_size 1 

  --per_device_eval_batch_size 1 

  --gradient_accumulation_steps 4 

  --learning_rate 1e-5 

  --eval_steps 100 

  --save_steps 100 

  --save_total_limit 2 

  --logging_steps 5 

  --max_length 8192 



  # ✅ 输出路径（保存训练 checkpoint）（路径需改）

  --output_dir /path/to/outputs/grpo 



  --warmup_ratio 0.05 

  --dataloader_num_workers 4 

  --dataset_num_proc 4 

  --num_generations 4 

  --temperature 1.0 

  --top_p 0.99 

  --top_k 50 

  --lazy_tokenize false 

  --system /path/to/prompt.txt 

  --deepspeed zero2 



  --log_completions true 



  # ✅ 保存完整训练日志（需改路径）

  2>&1 | tee /path/to/logs/log_file_grpo_3b.txt

