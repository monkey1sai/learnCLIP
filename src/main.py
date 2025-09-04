import os, math, json, random
from pathlib import Path
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Dataset
class ImageTextDataset(Dataset):
    def __init__(self, jsonl_path, image_root, tokenizer, max_len=40):
        self.items = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.items.append(json.loads(line))
        self.image_root = Path(image_root)
        self.tok = tokenizer
        self.max_len = max_len
        self.tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466,0.4578275,0.40821073),
                                 std=(0.26862954,0.26130258,0.27577711))
        ])

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        obj = self.items[idx]
        img = Image.open(self.image_root / obj["image_path"]).convert("RGB")
        pixel = self.tf(img)
        text = obj["text"]
        tok = self.tok(text, truncation=True, max_length=self.max_len,
                       padding='max_length', return_tensors='pt')
        return {
            "pixel_values": pixel,
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0)
        }

# 2. Vision Encoder (Tiny ViT from timm)
import timm
class VisionEncoder(nn.Module):
    def __init__(self, model_name="vit_tiny_patch16_224", proj_dim=256):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True)
        embed_dim = self.backbone.head.in_features
        self.backbone.reset_classifier(0)
        self.proj = nn.Linear(embed_dim, proj_dim)

    def forward(self, x):
        feats = self.backbone(x)  # (B, embed_dim)
        return self.proj(feats)   # (B, proj_dim)

# 3. Text Encoder (Tiny distilled model)
class TextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", proj_dim=256):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        hid = self.model.config.hidden_size
        self.proj = nn.Linear(hid, proj_dim)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:,0]  # CLS token
        return self.proj(cls)

# 4. CLIP-like Model
class MiniCLIP(nn.Module):
    def __init__(self, proj_dim=256, temperature_init=0.07):
        super().__init__()
        self.vision = VisionEncoder(proj_dim=proj_dim)
        self.text = TextEncoder(proj_dim=proj_dim)
        self.log_tau = nn.Parameter(torch.log(torch.tensor(temperature_init)))

    def forward(self, pixel_values, input_ids, attention_mask):
        v = self.vision(pixel_values)           # (B, D)
        t = self.text(input_ids, attention_mask)
        v = v / v.norm(dim=-1, keepdim=True)
        t = t / t.norm(dim=-1, keepdim=True)
        tau = torch.exp(self.log_tau)
        logits = (v @ t.t()) / tau
        return logits

# 5. Loss
def clip_loss(logits):
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i = nn.CrossEntropyLoss()(logits, labels)
    loss_t = nn.CrossEntropyLoss()(logits.t(), labels)
    return (loss_i + loss_t) / 2

# 6. Training Loop
def train(jsonl_path, image_root, epochs=5, batch_size=32, lr=1e-4):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    ds = ImageTextDataset(jsonl_path, image_root, tokenizer)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    model = MiniCLIP().to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dl:
            pixel = batch["pixel_values"].to(DEVICE)
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            logits = model(pixel, ids, mask)
            loss = clip_loss(logits)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(dl):.4f}")
    torch.save(model.state_dict(), "mini_clip.pt")

if __name__ == "__main__":
    # 假設 data/jsonl/train.jsonl 與 data/images/
    train("data/jsonl/train.jsonl", "data/images")
