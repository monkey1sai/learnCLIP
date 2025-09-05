
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BERT_MODEL_NAME = "hfl/chinese-roberta-wwm-ext"


class TextModel(nn.Module):
    def __init__(self, model: BertModel=BertModel.from_pretrained(BERT_MODEL_NAME)):
        super(TextModel, self).__init__()
        self.model = model.to(device)
        
    def forward(self, **inputs: dict):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        # # 如果需要 attention weights
        # 前提是你在模型 forward 時加上 output_attentions=True
        attention = outputs.attentions if "attentions" in outputs else None
        # pooler_output 是 BERT 預訓練時用來做分類的向量 (整句話的特徵) )
        # 比 last_hidden_state 多了一層 pooler tanh 激活函數
        pooler_output = outputs.pooler_output if "pooler_output" in outputs else None
        # outputs.last_hidden_state[:, 0, :] # 取得 [CLS] token 的向量（整句話的摘要特徵）
        return outputs.last_hidden_state[:, 0, :]
    
class Tokenizer:
    def __init__(self, pretrained_model_name=BERT_MODEL_NAME):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        
    def __call__(self, texts, max_length=128): # 支援單句或多句 max_length表示句子最大長度
        return self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

# 測試
if __name__ == "__main__":
    tokenizer = Tokenizer()
    text_model = TextModel()

    sentence = "今天天氣很好"
    inputs = tokenizer(sentence)
    outputs = text_model(**inputs)  # 丟進模型取得特徵向量
    print(outputs)  # 取得 [CLS] token 的向量