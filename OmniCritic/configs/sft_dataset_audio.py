import os
import json
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.build_prompt_audio import build_prompt  # 假设该模块已放置在 utils 下

# ========== 路径设置 ==========
audio_base_dir = "/home/Userlist/kongzicheng/omni-critic-r1/Clotho-AQA dataset/audio_files/audio_files"
json_path = "/home/Userlist/kongzicheng/omni-critic-r1/selected preference dataset/audio/filtered_audio_audio_omni.json"

output_dir = "/home/Userlist/kongzicheng/omni-critic-r1/OmniCritic-SFT/sft_dataset/audio/qwen2audio_qwen2.5omni"
output_train_jsonl = os.path.join(output_dir, "filtered_train.jsonl")
output_val_jsonl = os.path.join(output_dir, "filtered_val.jsonl")

# ========== 加载 JSON 数据 ==========
with open(json_path, "r", encoding="utf-8") as f:
    json_data = json.load(f)
print(f"✅ Loaded {len(json_data)} audio preference samples")

# ========== 加载分词器 ==========
tokenizer = AutoTokenizer.from_pretrained("/home/Userlist/kongzicheng/omni-critic-r1/Qwen2.5-Omni-3B/qwen/Qwen2.5-Omni-3B")
print("✅ Tokenizer loaded.")

# ========== 筛选数据（评分差大于等于 2） ==========
filtered_samples = []

for entry in tqdm(json_data, desc="Filtering audio samples"):
    score_A = entry.get("score_A", 0)
    score_B = entry.get("score_B", 0)
    
    # 计算评分差异
    if abs(score_A - score_B) >= 2:
        # 筛选符合条件的样本
        audio_path = os.path.join(audio_base_dir, entry["file_name"])
        if not os.path.exists(audio_path):
            print(f"⚠️ Audio not found: {audio_path}")
            continue
        
        # 构建prompt（音频特定任务）
        prompt = build_prompt(audio_path, entry["question"], entry["qwen2audio7b"], entry["qwen2.5omni7b"]).strip()

        # 直接构建 JSON 对象而不是字符串
        output = {
            "score_A": score_A,
            "score_B": score_B,
            "better": entry["better"],
            "reasoning": entry["reasoning"].strip(),
            "final_verdict": entry["final_verdict"].strip()
        }

        # 将符合条件的样本存入列表
        filtered_samples.append({
            "input": prompt,
            "output": output,  # 现在是 JSON 对象
            "audio": audio_path  # 用于后续加载或 SFT 框架处理
        })

print(f"✅ Filtered {len(filtered_samples)} audio samples based on score difference.")

# ========== 数据划分 ==========
split_idx = int(0.8 * len(filtered_samples))
train_set = filtered_samples[:split_idx]
val_set = filtered_samples[split_idx:]
print(f"✅ Split into {len(train_set)} train / {len(val_set)} val")

# ========== 保存为 JSONL ==========
def save_jsonl(path, dataset):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

# 保存训练集和验证集数据
save_jsonl(output_train_jsonl, train_set)
save_jsonl(output_val_jsonl, val_set)

print(f"✅ Saved: {output_train_jsonl}")
print(f"✅ Saved: {output_val_jsonl}")

