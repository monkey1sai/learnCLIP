from transformers import BertTokenizer, BertModel
from torchvision import models, transforms

import torch.nn as nn
import torch

from torchvision.models import resnet50, ResNet50_Weights


# =========
# 文字模型, BERT 預訓練的中文模型以及文字分詞器
# 將句子轉成 BERT 的輸入格式
# 這裡使用的是 hfl/chinese-roberta-wwm-ext 模型
# https://huggingface.co/hfl/chinese-roberta-wwm-ext
# =========
text_model = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

# 使用範例
sentence = "今天天氣很好"
inputs = tokenizer(sentence, return_tensors="pt")
outputs = text_model(**inputs) # 丟進模型取得特徵向
last_hidden_state = outputs.last_hidden_state  # 取得最後一層的隱藏狀態（每個 token 的向量）shape: [batch, seq_len, hidden_dim]
cls_embedding = last_hidden_state[:, 0, :]  # 取得 [CLS] token 的向量（整句特徵）此時的 batch size 是 1, shape: [batch, hidden_dim] , 0代表
print(cls_embedding) # 取得 [CLS] token 的向量（整句特徵）

# [CLS] 向量 [1, 768]
text_vec = outputs.last_hidden_state[:, 0, :] 

"""
    你可以把 cls_embedding 拿去做分類、相似度計算、聚類等。
    若要 fine-tune，需加上分類頭（如 nn.Linear）並用標註資料訓練。
"""
sentences = ["今天天氣很好", "我喜歡學習AI"]
inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
outputs = text_model(**inputs)
attention = outputs.attentions if "attentions" in outputs else None         # 前提是你在模型 forward 時加上 output_attentions=True
pooler_output = outputs.pooler_output if "pooler_output" in outputs else None # pooler_output 是 BERT 預訓練時用來做分類的向量




# =========
# 圖像模型
# =========
from PIL import Image
image_model = resnet50(weights=ResNet50_Weights.DEFAULT)
image_model.fc = nn.Identity()  # 拿掉最後分類層，輸出特徵向量

# 使用範例
img = Image.open("images/dog.jpg")
preprocess = transforms.Compose([
    transforms.Resize(256), # 縮放圖片最短邊到 256
    transforms.CenterCrop(224), # 裁切成 224x224
    transforms.ToTensor(), # 把 PIL 圖片轉成 PyTorch tensor，並把像素值縮放到 [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # 對每個通道（RGB）做標準化，讓資料分布和模型訓練時一致（這是 ImageNet 的平均值和標準差）
])


img_tensor = preprocess(img).unsqueeze(0)  # 增加 batch dimension
# .unsqueeze(0)：把 shape 從 [3, 224, 224] 變成 [1, 3, 224, 224]，多加一個 batch 維度，讓模型能一次處理多張圖（這裡是 1 張

with torch.no_grad():
    img_vec = image_model(img_tensor)

img_vec # shape: [1, 2048]
print(img_vec) # shape: [1, 2048]

# =========
# 融合 + 分類器
# =========
class MultiModalClassifier(nn.Module):
    def __init__(self, text_dim=768, img_dim=2048, hidden_dim=512, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(text_dim + img_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, text_vec, img_vec):
        combined = torch.cat((text_vec, img_vec), dim=1)
        x = self.fc1(combined)
        x = self.relu(x)
        return self.fc2(x)

model = MultiModalClassifier()

output = model(text_vec, img_vec)  # 得到 logits，shape: [1, num_classes]
print("分類結果 logits:", output)