#!/usr/bin/env python3
# -*- coding: utf‑8 -*-
"""
统计 merged_train.json 和 merged_test.json 中每条记录的 token 数量，并输出结果。
"""

import json
from transformers import AutoTokenizer
from tqdm import tqdm

# 指定 tokenizer 路径
model_path = "/nfs/mazhaoyu/llms/Qwen/Qwen3-8B"

# 加载 tokenizer，仅需占用非常少的资源
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 需要处理的两个文件
files = {
    "train": "/nfs/mazhaoyu/LLaMA-Factory/data/merged_train.json",
    "test":  "/nfs/mazhaoyu/LLaMA-Factory/data/merged_test.json",
}

# 结果字典，用于存储统计信息
stats = {}

for split, path in files.items():
    token_counts = []
    total_tokens = 0
    num_samples = 0
    mz32k = 0
    # 加载 JSON 文件（假设每行一条 JSON 或整个文件为 list）
    with open(path, "r", encoding="utf‑8") as f:
        data = json.load(f)
        if isinstance(data, dict):
            # 如果是 dict 包含 key: list
            data = data.get(split, data.get("items", []))
        elif not isinstance(data, list):
            raise ValueError(f"无法识别的 JSON 格式：{path}")
    
    # 遍历所有文本字段
    for rec in tqdm(data, desc=f"Tokenizing {split}", unit="rec"):
        text = rec.get("text") or rec.get("input") or rec.get("instruction") or rec.get("content")
        if text is None:
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        cnt = len(tokens)
        if cnt>32768:
            mz32k+=1
        token_counts.append(cnt)
        total_tokens += cnt
        num_samples += 1
    
    avg = total_tokens / max(num_samples, 1)
    stats[split] = {
        "samples": num_samples,
        "total_tokens": total_tokens,
        "avg_tokens": avg,
        "token_counts": token_counts,
        "mz32k": mz32k,
    }

# 输出结果
print("📊 Tokenization 统计结果：")
for split, st in stats.items():
    print(f"  • {split}: 大于32k:{st['mz32k']}样本数={st['samples']},  平均每条={st['avg_tokens']:.2f}")
