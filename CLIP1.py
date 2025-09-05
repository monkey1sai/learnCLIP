# from transformers import CLIPModel, CLIPProcessor

# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

# print("✅ 環境 OK，CLIP 模型載入完成！")



from PIL import Image
import requests

# 載入 CLIP
from transformers import CLIPProcessor, CLIPModel
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("✅ 環境 OK，CLIP 模型載入完成！")


# 測試圖片
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cat.png"
image = Image.open(requests.get(url, stream=True).raw)


# 輸入文字in
inputs = clip_processor(text=["a cat", "a dog"], images=image, return_tensors="pt", padding=True)

# 推論
outputs = clip_model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

print("分類結果：", probs)