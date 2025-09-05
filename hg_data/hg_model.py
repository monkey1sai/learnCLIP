from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


# 載入你訓練好的模型
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = AutoModelForSequenceClassification.from_pretrained("out/checkpoint-157")  # 這裡的 "out" 是你訓練時的 output_dir
model = model.to(device) # 移到 GPU (如果有的話)

# 測試新句子
text = "This movie is fantastic!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
inputs = {k: v.to(device) for k, v in inputs.items()} # 移到 GPU (如果有的話)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).item()
print("預測結果:", "正面" if pred == 1 else "負面")


"""
import torch

# ...existing code...

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).item()
print("預測結果:", "正面" if pred == 1 else "負面")

"""