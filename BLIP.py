from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

# 載入 BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 測試圖片
url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cat.png"
image = Image.open(requests.get(url, stream=True).raw)

# 處理圖片
inputs = processor(images=image, return_tensors="pt")

# 生成描述
out = model.generate(**inputs)
print("圖片描述：", processor.decode(out[0], skip_special_tokens=True))