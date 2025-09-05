from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


# 載入模型
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 多張圖片和多個文字描述
images = [Image.open("images/dog.jpg"), Image.open("images/cat.jpg")]
texts = ["dog", "cat", "bird", "car", "tree"]

# 轉成向量
inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
for k in inputs:
    inputs[k] = inputs[k].to(device)

outputs = model(**inputs)
image_embeds = outputs.image_embeds
text_embeds = outputs.text_embeds


# 計算相似度
similarity = torch.matmul(image_embeds, text_embeds.T)  # shape: [num_images, num_texts]
probs = similarity.softmax(dim=1)  # 每張圖片對所有文字的機率分布

threshold = 0.5  # 設定一個閾值
# 找出每張圖片最相近的文字描述
for i, img in enumerate(images):
    pred_idx = torch.argmax(probs[i]).item() # pred_idx 是最相近的文字描述的索引
    pred_label = texts[pred_idx] # 用索引找出文字描述
    pred_score = probs[i, pred_idx].item()  # 第i張圖片對應到 pred_idx 文字的分數
    print(f"所有分數: {[f'{t}:{probs[i, j]:.4f}' for j, t in enumerate(texts)]}")
    print(f"圖片{i+1}最相近的描述: {pred_label} (分數: {pred_score:.4f})")
    if pred_score > threshold:
        print(f"圖片{i+1}屬於: {pred_label} (分數: {pred_score:.4f})")
    else:
        print(f"圖片{i+1}無明確類別 (最高分數: {pred_score:.4f})")
    print('---')

