import torchvision.transforms as transforms
import torch
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
    
"""image standardization: 對每個通道RGB做標準,讓資料分布和模型訓練時一致
    這是 ImageNet 的平均值和標準差 """
class ImageProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(256), # 縮放圖片最短邊到 256
            transforms.CenterCrop(224), # 裁切成 224x224
            transforms.ToTensor(), # 把 PIL 圖片轉成 PyTorch tensor，並把像素值縮放到 [0, 1]
            # 對每個通道（RGB）做標準化，讓資料分布和模型訓練時一致（這是 ImageNet 的平均值和標準差）
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    """__call__ 讓物件可以像函數一樣被呼叫"""
    def __call__(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        # .unsqueeze(0)：把 shape 從 [3, 224, 224] 變成 [1, 3, 224, 224]，多加一個 batch 維度，讓模型能一次處理多張圖（這裡是 1 張）
        return self.transform(image).unsqueeze(0)


""" 經過 ImageProcessor 處理後的圖片張量,丟進 ImageModel 取得特徵向量 """
class ImageModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module= resnet50(weights=ResNet50_Weights.DEFAULT)):
        super().__init__()
        self.model = model
        # 破壞RN50最後的分類層,因此只輸出特徵向量
        self.model.fc = torch.nn.Identity() # 用 nn.Identity() 取代最後分類層,因此只輸出特徵向量

    """forward 丟進模型取得特徵向量"""
    def forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        img_vec = self.model(img_tensor)
        return img_vec  # 將 image 轉到特徵向量 [1, 2048]
    
if __name__ == "__main__":
    from torchvision.models import resnet50, ResNet50_Weights
    import requests
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 測試 ImageProcessor 和 ImageModel
    image_processor = ImageProcessor()
    image_model = ImageModel()
    image_model = image_model.to(device)

    # 測試圖片
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cat.png"
    image = Image.open(requests.get(url, stream=True).raw)


    # 推論
    image_model.eval()
    with torch.no_grad():
        # 處理圖片標準化符合模型需求
        img_tensor = image_processor(image)
        # 取得圖片特徵向量
        img_tensor = img_tensor.to(device)
        img_vec = image_model(img_tensor)
        print('img_vec:', img_vec)  # shape: [1, 2048]
    
    # 訓練
    image_model.train()
    for _ in range(2):  # 模擬兩個 batch
        img_tensor = image_processor(image)
        img_tensor = img_tensor.to(device)
        img_vec = image_model(img_tensor)
        print('img_vec:', img_vec)  # shape: [1, 2048]
        # 這裡可以加上損失計算和反向傳播等訓練步驟
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
    print("✅ ImageProcessor 和 ImageModel 測試完成！")