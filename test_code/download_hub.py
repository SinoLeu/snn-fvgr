from huggingface_hub import hf_hub_download
import timm
import torch

# 下载模型权重文件
# model_path = 's_cars_pytorch_model.bin'

# 创建 timm 的 resnet50 模型
model = timm.create_model("resnet50.a1_in1k", pretrained=True)

# 加载权重
# state_dict = torch.load(model_path, map_location="cpu")
# model.load_state_dict(state_dict)

# 设置为评估模式
# model.eval()

print("模型加载成功！")
