
# 安裝環境

```shell
安裝環境於
C:\Users\IOT>nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Fri_Jun_14_16:44:19_Pacific_Daylight_Time_2024
Cuda compilation tools, release 12.6, V12.6.20
Build cuda_12.6.r12.6/compiler.34431801_0

C:\Users\IOT>nvidia-smi
Thu Sep  4 11:15:06 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.97                 Driver Version: 580.97         CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060 Ti   WDDM  |   00000000:01:00.0  On |                  N/A |
|  0%   43C    P8              6W /  160W |    1412MiB /   8188MiB |      2%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+


conda create -n  mutil.311 python=3.11 -y
conda activate mutil.311
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install transformers datasets accelerate bitsandbytes sentencepiece timm wandb lightning pillow numpy scikit-learn

```
