import os
import random
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 모델 로드
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/Janus-Pro-7B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/Janus-Pro-7B")

def generate_pose_explanation(src, tgt):
    prompt = f"Describe the movement needed to transition from pose {src} to pose {tgt}."
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 이미지 폴더 경로
IMAGE_DIR = "images"
OUTPUT_CSV = "pose_explanations.csv"

# 이미지 파일 리스트 가져오기
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))])
num_images = len(image_files)

# 모든 이미지에 대해 99개의 랜덤한 비교 대상 선택
data_pairs = []
for img in image_files:
    other_images = [x for x in image_files if x != img]
    random.shuffle(other_images)
    selected_images = other_images[:99]
    
    for other_img in selected_images:
        data_pairs.append((img, other_img))
        data_pairs.append((other_img, img))  # 역순 추가

# 결과 저장
output_data = []
for src, tgt in data_pairs:
    explanation = generate_pose_explanation(src, tgt)
    output_data.append([src, tgt, explanation])

# CSV 파일 저장
with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["source", "target", "pose_explanation"])
    writer.writerows(output_data)

print(f"Generated {len(output_data)} pose explanation pairs and saved to {OUTPUT_CSV}")
