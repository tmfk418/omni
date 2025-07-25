\# OmniCritic-R1



本项目包含 OmniCritic-R1 的完整代码与数据，用于复现论文中的多模态奖励模型训练与评估流程。



\## 🔧 环境准备



建议使用以下环境配置：



\- Python >= 3.10

\- CUDA >= 11.8

\- 推荐使用 Linux + 多卡 GPU



\## 1. 创建并激活环境、下载依赖和模型：



conda create -n omnicritic python=3.10 -y

conda activate omnicritic

pip install -r requirements.txt

cd omniR1-sft-master/ms-swift-main

pip install -e .





下载qwen2.5omni3b和7b

以3b为例

from modelscope.hub.snapshot\_download import snapshot\_download



model\_dir = snapshot\_download(

&nbsp;   'qwen/Qwen2.5-Omni-3B',

&nbsp;   cache\_dir='./Qwen2.5-Omni-3B',  # ✅ 可指定本地路径

&nbsp;   revision='master'

)





\##2. 克隆本项目

在你希望存放的路径下执行：

git clone https://github.com/tmfk418/omni.git

cd omni



\##3. 数据准备



需要下载的数据集托管在 Hugging Face 上。可以使用 huggingface\_hub 库进行下载。



安装 huggingface\_hub 库：



pip install huggingface\_hub



下载数据集：



from huggingface\_hub import hf\_hub\_download



下载音频、视频和图像数据集



image\_dataset = hf\_hub\_download("TMFK/omnir1-dataset", "rlaif-v-dataset.tar.gz")



video\_dataset = hf\_hub\_download("TMFK/omnir1-dataset", "C:\\Users\\tmfk1\\video\\video.zip")



audio\_dataset = hf\_hub\_download("TMFK/omnir1-dataset", "Clotho-AQA dataset.zip")



\##4.更新sft和rl的数据集路径

运行以下脚本来更新音频和视频路径：



更新路径



python configs/change\_path.py



确保路径与您下载的文件相匹配。脚本会根据指定的基础路径更新 JSONL 文件中的sft和rl图像、音频和视频路径。

辛苦转换后看一下是不是和实际的数据集的路径相符



\##5. 运行sft代码

bash scripts/sft\_3b.sh

bash scripts/sft\_3b\_lora.sh

bash scripts/sft\_7b.sh

bash scripts/sft\_7b\_lora.sh



注意修改里面的所有路径

以sft\_3b.sh为例
 ✅ 根据实际显卡数和资源配置修改

export CUDA\_VISIBLE\_DEVICES=0,1,2,3        # ← ✅ 若用单卡或不同编号需修改

✅ 可选项，根据任务需要设置（图像/视频/音频）

export MAX\_PIXELS=262144

export FPS=1

export FPS\_MAX\_FRAMES=20

export VIDEO\_MAX\_PIXELS=32768

export AUDIO\_MEMMAP=true

export AUDIO\_CHUNK\_SIZE=4000

export AUDIO\_NUM\_WORKERS=4

export VIDEO\_READER\_BACKEND=torchvision

export TOKENIZERS\_PARALLELISM=false

export OMP\_NUM\_THREADS=1

export SWIFT\_DEVICE\_MAP=auto

✅ 启动训练主命令（根据模型、路径修改）

torchrun --nproc\_per\_node=4 \\                        # ← ✅ 改为你的显卡数（如 1、2、4、8）

&nbsp; /data/yiwei.ru/omniR1-sft-master/ms-swift-main/swift/cli/sft.py \\



&nbsp; --do\_train \\



&nbsp; # ✅ 模型路径：修改为你本地/远程的基座模型路径

&nbsp; --model /data/yiwei.ru/modelscope/Qwen/Qwen2.5-Omni-3B \\        # ← ✅ 替换为你使用的模型路径



&nbsp; # ✅ 数据集路径：使用你准备的数据集（建议绝对路径）

&nbsp; --dataset \\

&nbsp;   /data/yiwei.ru/omniR1-sft-master/OmniCritic/sft\_dataset/audio/final\_sft\_data.jsonl \\

&nbsp;   /data/yiwei.ru/omniR1-sft-master/OmniCritic/sft\_dataset/video/final\_sft\_data.jsonl \\

&nbsp;   /data/yiwei.ru/omniR1-sft-master/OmniCritic/sft\_dataset/image/final\_sft\_data.jsonl \\     # ← ✅ 替换为你准备的 jsonl 文件



&nbsp; --train\_type full \\                     



&nbsp; # ✅ 输出路径：训练结果会保存在这里

&nbsp; --output\_dir /data/yiwei.ru/omniR1-sft-master/OmniCritic/outputs \\   # ← ✅ 根据你想保存的位置修改



&nbsp; # ✅ 训练参数（辛苦根据实际情况调整）

&nbsp; --num\_train\_epochs 2 \\

&nbsp; --per\_device\_train\_batch\_size 1 \\

&nbsp; --per\_device\_eval\_batch\_size 1 \\

&nbsp; --gradient\_accumulation\_steps 12 \\

&nbsp; --learning\_rate 5e-6 \\

&nbsp; --warmup\_ratio 0.03 \\

&nbsp; --max\_length 3072 \\

&nbsp; --eval\_steps 100 \\

&nbsp; --save\_steps 100 \\

&nbsp; --save\_total\_limit 4 \\

&nbsp; --logging\_steps 5 \\

&nbsp; --dataloader\_num\_workers 2 \\

&nbsp; --torch\_dtype bfloat16 \\               

&nbsp; --deepspeed zero2 \\



&nbsp; # ✅ 输出日志（建议修改路径）

&nbsp; 2>\&1 | tee /data/yiwei.ru/omniR1-sft-master/OmniCritic/logs/log\_file\_3b.txt   # ← ✅ 可替换为你本地的 logs 路径



\##6.运行grpo代码



bash scripts/grpo\_3b.sh

bash scripts/grpo\_7b.sh



export MAX\_PIXELS=262144                     

export VIDEO\_MAX\_PIXELS=65536                

export NPROC\_PER\_NODE=4                       # ✅ 使用 GPU 数量（必须与 CUDA\_VISIBLE\_DEVICES 数量一致）

export CUDA\_VISIBLE\_DEVICES=0,1,2,3           # ✅ 设置实际可用显卡编号

export FPS=1

export FPS\_MAX\_FRAMES=20

export AUDIO\_MEMMAP=true

export AUDIO\_CHUNK\_SIZE=4000

export AUDIO\_NUM\_WORKERS=4



swift rlhf \\

&nbsp; --rlhf\_type grpo \\



&nbsp; # ✅ 模型路径（替换为你自己的预训练模型，如 3B 或 7B）

&nbsp; --model /data/yiwei.ru/modelscope/Qwen/Qwen2.5-Omni-3B \\

&nbsp; --reward\_funcs mm\_format mm\_content mm\_rubric \\

&nbsp; --reward\_weights 0.4 0.4 0.2 \\

&nbsp; --train\_type lora \\

&nbsp; --lora\_rank 8 \\

&nbsp; --lora\_alpha 32 \\

&nbsp; --target\_modules all-linear \\

&nbsp; --torch\_dtype bfloat16 \\                    

&nbsp; # ✅ 多模态 RL 数据路径（替换为你准备的 jsonl 文件）

&nbsp; --dataset \\

&nbsp;     /data/.../audio/final\_final\_rl\_data.jsonl \\

&nbsp;     /data/.../video/final\_final\_rl\_data.jsonl \\

&nbsp;     /data/.../image/final\_final\_rl\_data.jsonl \\



&nbsp; # ✅ 外部插件：自定义奖励函数 / 数据解析（已经放在ms—swift-main的里面了，只改路径前缀就好）

&nbsp; --external\_plugins /path/to/new\_plugin.py \\



&nbsp; # 模型生成与训练参数

&nbsp; --max\_completion\_length 2048 \\

&nbsp; --num\_train\_epochs 1 \\

&nbsp; --per\_device\_train\_batch\_size 1 \\

&nbsp; --per\_device\_eval\_batch\_size 1 \\

&nbsp; --gradient\_accumulation\_steps 4 \\

&nbsp; --learning\_rate 1e-5 \\

&nbsp; --eval\_steps 100 \\

&nbsp; --save\_steps 100 \\

&nbsp; --save\_total\_limit 2 \\

&nbsp; --logging\_steps 5 \\

&nbsp; --max\_length 8192 \\



&nbsp; # ✅ 输出路径（保存训练 checkpoint）（路径需改）

&nbsp; --output\_dir /path/to/outputs/grpo \\



&nbsp; --warmup\_ratio 0.05 \\

&nbsp; --dataloader\_num\_workers 4 \\

&nbsp; --dataset\_num\_proc 4 \\

&nbsp; --num\_generations 4 \\

&nbsp; --temperature 1.0 \\

&nbsp; --top\_p 0.99 \\

&nbsp; --top\_k 50 \\

&nbsp; --lazy\_tokenize false \\

&nbsp; --system /path/to/prompt.txt \\

&nbsp; --deepspeed zero2 \\



&nbsp; --log\_completions true \\



&nbsp; # ✅ 保存完整训练日志（需改路径）

&nbsp; 2>\&1 | tee /path/to/logs/log\_file\_grpo\_3b.txt

