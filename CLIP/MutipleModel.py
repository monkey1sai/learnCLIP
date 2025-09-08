import torch

class MutipleModel(torch.nn.Module):
    def __init__(self,  text_dim=768, img_dim=2048, hidden_dim=512, num_classes=2):
        super().__init__()
        self.fc1 = torch.nn.Linear(text_dim + img_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, text_vec, image_vec):
        combined = torch.cat((text_vec, image_vec), dim=1)  # 在特徵維度上串接
        
        # 把「融合後的多模態特徵」丟進一個全連接層（Linear），
        # 再經過 ReLU 激活函數，
        # 最後再丟進另一個全連接層得到分類結果
        
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    
    # 假設 text_vec 和 img_vec 是從 TextModel 和 ImageModel 得到的特徵向量
    text_vec = torch.randn(1, 768)   # 假設 batch size
    img_vec = torch.randn(1, 2048)  # 假設 batch size
    print("text_vec:", text_vec)
    print("img_vec: ", img_vec)
    
    
    # 推論
    model = MutipleModel().to(device)
    model.eval()
    with torch.no_grad():
        text_vec = text_vec.to(device)
        img_vec = img_vec.to(device)    
        output = model(text_vec, img_vec)  # 得到 logits，shape: [1, num_classes]

    # logits 是分類分數，通常用 softmax 轉成機率。
    print("分類結果 logits:", output)
    print("分類結果 logits shape:", output.shape)  # shape: [1, num_classes]
    probs = torch.nn.functional.softmax(output, dim=1)
    print("分類結果機率:", probs)
    print("分類結果機率 shape:", probs.shape)  # shape: [1, num_classes]
    # 假設標籤（假設是二分類，標籤為 1）
    label = torch.tensor([1]).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    for epoch in range(5):  # 訓練 5 次
        optimizer.zero_grad() # 清空梯度
        output = model(text_vec, img_vec)
        loss = criterion(output, label)
        loss.backward() # 反向傳播計算梯度
        optimizer.step()
        
        # loss 代表訓練過程中模型的學習狀況，越低越好。

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    