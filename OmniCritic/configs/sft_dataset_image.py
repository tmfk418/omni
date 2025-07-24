import os
import sys
import json
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm

# ✅ 添加这一行确保模块导入正常
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.build_prompt_image import build_prompt

# === 输入路径配置 ===
json_path = "/home/Userlist/kongzicheng/omni-critic-r1/selected preference dataset/image/doubao_image_llava_qwen_output.json"
train_arrow_path = "/home/Userlist/kongzicheng/omni-critic-r1/OmniCritic-SFT/sft_dataset/image/llava_qwen/train"
val_arrow_path = "/home/Userlist/kongzicheng/omni-critic-r1/OmniCritic-SFT/sft_dataset/image/llava_qwen/val"
output_train_jsonl = "/home/Userlist/kongzicheng/omni-critic-r1/OmniCritic-SFT/sft_dataset/image/llava_qwen/train.jsonl"
output_val_jsonl   = "/home/Userlist/kongzicheng/omni-critic-r1/OmniCritic-SFT/sft_dataset/image/llava_qwen/val.jsonl"

# === 加载标注数据 ===
with open(json_path, "r", encoding="utf-8") as f:
    json_data = json.load(f)

# === 加载 Arrow 图像数据并建立 image_path 索引 ===
train_arrow = load_from_disk(train_arrow_path)
val_arrow = load_from_disk(val_arrow_path)

image_index = {}
for ds in [train_arrow, val_arrow]:
    for i, path in enumerate(ds["image_path"]):
        image_index[path] = (ds, i)

# === 构造样本 ===
train_set, val_set = [], []

for entry in tqdm(json_data, desc="Building samples"):
    image_path = entry["image_path"]
    if image_path not in image_index:
        continue  # 跳过图像丢失的项

    question = entry["question"]
    answer_a = entry["llava1_5_vl_7b_output"]
    answer_b = entry["qwen2_5_vl_7b_output"]

    prompt = build_prompt(image_path, question, answer_a, answer_b).strip()

    output = json.dumps({
        "score_A": entry["score_A"],
        "score_B": entry["score_B"],
        "better": entry["better"],
        "reasoning": entry["reasoning"].strip(),
        "final_verdict": entry["final_verdict"].strip()
    }, ensure_ascii=False)

    ds, idx = image_index[image_path]
    sample = {
        "input": prompt,
        "output": output,
        "image": ds[idx]["image"]  # PIL.Image 对象
    }

    if ds == train_arrow:
        train_set.append(sample)
    else:
        val_set.append(sample)

# === 保存为 .jsonl 格式 ===
def save_jsonl(path, dataset):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps({
                "input": item["input"],
                "output": item["output"],
                "image": item["image"]
            }, ensure_ascii=False, default=str) + "\n")

save_jsonl(output_train_jsonl, train_set)
save_jsonl(output_val_jsonl, val_set)

print(f"✅ Saved: {len(train_set)} train samples to {output_train_jsonl}")
print(f"✅ Saved: {len(val_set)} validation samples to {output_val_jsonl}")
