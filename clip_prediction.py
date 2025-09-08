from CLIP.ImageVec import ImageProcessor, ImageModel
from CLIP.TextCLS import TextModel, Tokenizer
from CLIP.MutipleModel import MutipleModel

if __name__ == "__main__":
    import torch
    from PIL import Image
    import requests

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 測試 ImageProcessor 和 ImageModel
    image_processor = ImageProcessor()
    image_model = ImageModel()
    image_model = image_model.to(device)


    # 測試圖片特徵擷取
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cat.png"
    image = Image.open(requests.get(url, stream=True).raw)

    # 處理圖片
    img_tensor = image_processor(image)
    img_tensor = img_tensor.to(device)
    image_model.eval()
    with torch.no_grad():
        # 取得圖片特徵向量
        img_vec = image_model(img_tensor)
        print('img_vec:', img_vec)  # shape: [1, 2048]
        print('img_vec shape:', img_vec.shape) # shape: [1, 2048]  # shape: [1, 2048]
    
    
    # 測試文字特徵擷取
    text = "a cat"
    tokenizer = Tokenizer()
    text_model = TextModel()
    text_model = text_model.to(device)
    inputs = tokenizer(text)

    text_model.eval()
    with torch.no_grad():
        # 取得文字特徵向量
        inputs = {k: v.to(device) for k, v in inputs.items()}
        text_vec = text_model(**inputs)
        print('text_vec:', text_vec)  # shape: [1, 768]
        print('text_vec shape:', text_vec.shape) # shape: [1, 768]
        
    # 融合文字和圖片特徵向量
    model = MutipleModel()
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(text_vec, img_vec)
        print("分類結果 logits:", output)
        print("分類結果 logits shape:", output.shape)  # shape: [1, num_classes]
        results = torch.softmax(output, dim=1) # 針對類別維度轉成機率
        print("分類結果機率:", results)
        print("分類結果機率 shape:", results.shape)  # shape: [1, num_classes]
    print("✅ CLIP 多模態分類測試完成！")
    