from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests

# 載入模型
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 測試圖片
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cat.png"
image = Image.open(requests.get(url, stream=True).raw)

# 輸入文字 & 圖片
inputs = processor(text=["a cat", "a dog"], images=image, return_tensors="pt", padding=True)

# 推論
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)  

print("結果：", probs)
