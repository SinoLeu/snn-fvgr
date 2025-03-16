from torchvision import datasets, models, transforms
from torchvision import models, transforms
import torch

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

x = torch.rand(1,3,224,224)


best_model = torch.load("./logs/fine_tune_resnet50/version_8/checkpoints/best_model.ckpt")
print(best_model['state_dict'].keys())
y,mid_feature = model(x,return_mid=True)


print(y.shape,mid_feature[0].shape,mid_feature[1].shape,mid_feature[2].shape,mid_feature[3].shape,mid_feature[4].shape)
