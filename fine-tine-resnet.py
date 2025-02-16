import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
# import pytorch_lightning as pl
# from pytorch_lightning import Trainer
from torchvision import models, transforms
import torch
# from timm import create_model
from torchvision import transforms, datasets
import lightning as L
import timm
# import os
from torch.optim.lr_scheduler import StepLR

import pytorch_lightning as pl
# your favorite machine learning tracking tool
from pytorch_lightning.loggers import WandbLogger

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy

from torchvision import transforms
# from torchvision.datasets import StanfordCars
# from torchvision.datasets.utils import download_url
import torchvision.models as models
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


class StanfordCarsDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Augmentation policy for training set
        self.augmentation = transforms.Compose([
              transforms.RandomResizedCrop(size=300, scale=(0.8, 1.0)),
              transforms.RandomRotation(degrees=15),
              transforms.RandomHorizontalFlip(),
              transforms.CenterCrop(size=300),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        # Preprocessing steps applied to validation and test set.
        self.transform = transforms.Compose([
              # transforms.Resize(size=224),
              # # transforms.CenterCrop(size=224),
              # transforms.ToTensor(),
            transforms.Resize(300),
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        
        self.num_classes = 196

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # build dataset
        self.train = datasets.ImageFolder(root='../data/cars/train', transform=self.augmentation)
        # split dataset
        # self.train, self.val = random_split(dataset, [6500, 1644])

        # self.test = StanfordCars(root=self.data_dir, download=True, split="test")
        
        self.test = datasets.ImageFolder(root='../data/cars/test', transform=self.transform)

        # self.train.dataset.transform = self.augmentation
        # self.val.dataset.transform = self.transform
        # self.test.dataset.transform = self.transform
        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=14)

    def val_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=14)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=14)


class LitModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=0.1, transfer=True):
        super().__init__()
        
        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.dim = input_shape
        self.num_classes = num_classes

        
        # transfer learning if pretrained=True
        self.feature_extractor = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        # self.feature_extractor = timm.create_model("resnet50", pretrained=True)
        in_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(in_features, num_classes)
        # self.feature_extractor.torch.load(model_path, map_location="cpu")
        # 加载权重
        # model_path = 's_cars_pytorch_model.bin'
        # state_dict = torch.load(model_path)
        # self.feature_extractor.load_state_dict(state_dict)
        # self.feature_extractor.fc = nn.Identity()
        # for name, param in self.feature_extractor.named_parameters():
        #     if "fc" not in name:  # 冻结所有非 fc 层的参数
        #         param.requires_grad = False

        self.classifier = nn.Identity()
        # self.classifier = nn.Sequential(
        # #     Dense(1024, activation = 'relu'),
        # # Dropout(0.25),
        # # Dense(512, activation = 'relu'),
        # # Dropout(0.25),
        # # Dense(196, activation = 'softmax')
        #     nn.Linear(in_features, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(0.25),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.25),
        #     nn.Linear(512, num_classes),
        #     # nn.ReLU(),
            
        #     # nn.Linear(512, num_classes)
        # )
        # self.classifier = nn.Sequential(
        #     nn.Linear(in_features, 512),  # 先降维
        #     nn.ReLU(),                    # 加入激活函数
        #     nn.Dropout(p=0.2),    # 防止过拟合
        #     nn.Linear(512, num_classes)    # 输出类别
        # )
        # if transfer:
        #     # layers are frozen by using eval()
        #     self.feature_extractor.eval()
        #     # freeze params
        #     for param in self.feature_extractor.parameters():
        #         param.requires_grad = False
        
        # n_sizes = self._get_conv_output(input_shape)

        

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass",num_classes=196)
  
    # returns the size of the output tensor going into the Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(tmp_input) 
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
        
    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        x = self.feature_extractor(x)
        # print(x.shape)
        return x
    
    # will be used during inference
    def forward(self, x):
       x = self._forward_features(x)
       x = x.view(x.size(0), -1)
       x = self.classifier(x)
       
       return x
    
    def training_step(self, batch):
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        loss = self.criterion(out, gt)

        acc = self.accuracy(out, gt)

        self.log("train/loss", loss)
        self.log("train/acc", acc)

        return loss
    
    def validation_step(self, batch, batch_idx):
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        loss = self.criterion(out, gt)

        self.log("val/loss", loss)

        acc = self.accuracy(out, gt)
        self.log("val/acc", acc)

        return loss
    
    def test_step(self, batch, batch_idx):
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        loss = self.criterion(out, gt)
        
        return {"loss": loss, "outputs": out, "gt": gt}
    
    def test_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        output = torch.cat([x['outputs'] for x in outputs], dim=0)
        
        gts = torch.cat([x['gt'] for x in outputs], dim=0)
        
        self.log("test/loss", loss)
        acc = self.accuracy(output, gts)
        self.log("test/acc", acc)
        
        self.test_gts = gts
        self.test_output = output
    
    def configure_optimizers(self):
        # return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=0)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'  # 适用于 ReduceLROnPlateau
        }

# # 设置 TensorBoardLogger
# logger = TensorBoardLogger("logs", name="TransferLearning")
from pytorch_lightning.loggers import CSVLogger

logger = CSVLogger("logs", name="fine_tune_resnet101")

checkpoint_callback = ModelCheckpoint(
    monitor="val/acc",           # 监控验证集准确率
    mode="max",                  # 追踪最大值
    save_top_k=1,                # 保存最佳模型
    verbose=True,                # 输出日志
    filename="best_model"        # 文件名
)
## 1

dm = StanfordCarsDataModule(batch_size=64)
model = LitModel((3, 300, 300), 196, transfer=True)
## strategy="ddp", num_nodes=4
trainer = pl.Trainer(logger=logger, max_epochs=150, accelerator="gpu",callbacks=[checkpoint_callback],strategy="ddp")

trainer.fit(model, dm)
print("end....")