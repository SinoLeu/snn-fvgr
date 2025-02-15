import torch
import timm

## HF_ENDPOINT=https://hf-mirror.com python download_huggface.py
# 加载预训练的 ResNet-50 (a3_in1k)
model = timm.create_model("resnet50", pretrained=True)

# 切换为评估模式
model.eval()

# 打印模型结构
print(model)

