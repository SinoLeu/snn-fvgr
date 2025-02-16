from torchvision import datasets, models, transforms
from torchvision import models, transforms
import torch

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

x = torch.rand(1,3,224,224)

y,mid_feature = model(x,return_mid=True)

print(y.shape,mid_feature[0].shape,mid_feature[1].shape,mid_feature[2].shape,mid_feature[3].shape,mid_feature[4].shape)
