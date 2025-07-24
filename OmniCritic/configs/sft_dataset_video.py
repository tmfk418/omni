import os
import sys
import json
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer

# 添加 utils 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.build_prompt_video import build_prompt  # 确保 utils 下有此函数

# ========== 路径设置 ==========
video_base_dir = "/home/kongzicheng"
json_path = "/home/Userlist/kongzicheng/omni-critic-r1/selected preference dataset/video/doubao_video_output.json"

output_dir = "/home/Userlist/kongzicheng/omni-critic-r1/OmniCritic-SFT/sft_dataset/video"
output_train_jsonl = os.path.join(output_dir, "filtered_train.jsonl")
output_val_jsonl = os.path.join(output_dir, "filtered_val.jsonl")

# ========== 加载 JSON 数据 ==========
with open(json_path, "r", encoding="utf-8") as f:
    json_data = json.load(f)
print(f"✅ Loaded {len(json_data)} video preference samples")

# ========== 加载分词器 ==========
tokenizer = AutoTokenizer.from_pretrained("/home/Userlist/kongzicheng/omni-critic-r1/Qwen2.5-Omni-3B/qwen/Qwen2.5-Omni-3B")
print("✅ Tokenizer loaded.")

# ========== 构造样本并筛选 ==========
samples = []

for entry in tqdm(json_data, desc="Filtering and building video samples"):
    score_A = entry.get("score_A", 0)
    score_B = entry.get("score_B", 0)

    # 筛选评分差异大于等于 2 的样本
    if abs(score_A - score_B) < 2:
        continue

    video_path = os.path.join(video_base_dir, entry["video_path"])
    if not os.path.exists(video_path):
        print(f"⚠️ Video not found: {video_path}")
        continue

    prompt = build_prompt(video_path, entry["question"], entry["qwen2_5_vl_3b_output"], entry["qwen2_5_vl_7b_output"]).strip()

    output = {
        "score_A": score_A,
        "score_B": score_B,
        "better": entry["better"],
        "reasoning": entry["reasoning"].strip(),
        "final_verdict": entry["final_verdict"].strip()
    }

    samples.append({
        "input": prompt,
        "output": output,
        "video": video_path
    })

print(f"✅ Filtered and built {len(samples)} video samples")

# ========== 数据划分 ==========
split_idx = int(0.8 * len(samples))
train_set = samples[:split_idx]
val_set = samples[split_idx:]
print(f"✅ Split into {len(train_set)} train / {len(val_set)} val")

# ========== 保存为 JSONL ==========
def save_jsonl(path, dataset):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

save_jsonl(output_train_jsonl, train_set)
save_jsonl(output_val_jsonl, val_set)

print(f"✅ Saved: {output_train_jsonl}")
print(f"✅ Saved: {output_val_jsonl}")
