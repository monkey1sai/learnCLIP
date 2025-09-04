
# 安裝環境
1. conda create -n  mutil.311 python=3.11 -y
2. conda activate mutil.311
3. pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
4. pip install transformers datasets accelerate bitsandbytes sentencepiece timm wandb lightning pillow numpy scikit-learn
