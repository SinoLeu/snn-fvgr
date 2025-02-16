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
import argparse
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
from pytorch_lightning.loggers import CSVLogger

## python fine-tine-resnet_ann.py --batch_size 64 --learning_rate 0.1 --input_size 300 --train_dir '../data/cars/train' --test_dir '../data/cars/test' --resnet_scale '50' --max_epochs 150  --checkpoint_dir 'logs' --num_classes 196 --is_distributed  --is_transfer

## fine-tune resnet
class StanfordCarsDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_dir: str = './',test_dir: str = './',input_size:int=300,num_class:int = 196):
        super().__init__()
        # self.data_dir = data_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_classes = num_class

        # Augmentation policy for training set
        self.augmentation = transforms.Compose([
              transforms.RandomResizedCrop(size=self.input_size, scale=(0.8, 1.0)),
              transforms.RandomRotation(degrees=15),
              transforms.RandomHorizontalFlip(),
              transforms.CenterCrop(size=self.input_size),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        # Preprocessing steps applied to validation and test set.
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        
        # self.num_classes = 196

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # build dataset '../data/cars/train'
        self.train = datasets.ImageFolder(root=self.train_dir, transform=self.augmentation)
        self.test = datasets.ImageFolder(root=self.test_dir, transform=self.transform)
        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=14)

    def val_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=14)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=14)


class LitModel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=0.1, transfer=True,resnet_scale:str = '50'):
        super().__init__()
        
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        
        ## 
        if resnet_scale == '18':
            if transfer:
                self.feature_extractor = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.feature_extractor = models.resnet18(weights=None)
        elif resnet_scale == '50':
            if transfer:
                self.feature_extractor = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                self.feature_extractor = models.resnet50(weights=None)
        elif resnet_scale == '101':
            if transfer:
                self.feature_extractor = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            else:
                self.feature_extractor = models.resnet101(weights=None)
        elif resnet_scale == '34':
            if transfer:
                self.feature_extractor = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            else:
                self.feature_extractor = models.resnet34(weights=None)
            
        in_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(in_features, num_classes)
        self.classifier = nn.Identity()
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass",num_classes=self.num_classes)
  
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
            'monitor': 'val_loss'
        }

# # 设置 TensorBoardLogger

def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet model on Stanford Cars")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=0.1, help="Learning rate")
    parser.add_argument('--input_size', type=int, default=300, help="Input size for images")
    parser.add_argument('--train_dir', type=str, default='./train', help="Directory for training data")
    parser.add_argument('--test_dir', type=str, default='./test', help="Directory for testing data")
    parser.add_argument('--resnet_scale', type=str, default='50', choices=['18', '34', '50', '101'], help="ResNet scale")
    parser.add_argument('--max_epochs', type=int, default=150, help="Number of epochs for training")
    parser.add_argument('--checkpoint_dir', type=str, default="logs", help="Directory to save checkpoints")
    parser.add_argument('--num_classes', type=int, default=196, help="classificer classes")
    parser.add_argument('--is_distributed', action='store_true', help="Enable distributed training")
    parser.add_argument('--is_transfer', action='store_true', help="Enable distributed training")
    return parser.parse_args()


def main():
    args = parse_args()
    logger_name = f"fine_tune_resnet{args.resnet_scale}"
    logger = CSVLogger(args.checkpoint_dir, name=logger_name)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/acc",           # 监控验证集准确率
        mode="max",                  # 追踪最大值
        save_top_k=1,                # 保存最佳模型
        verbose=True,                # 输出日志
        filename="best_model"        # 文件名
    )
    ## 1
    dm = StanfordCarsDataModule(batch_size=args.batch_size, train_dir=args.train_dir, test_dir=args.test_dir, input_size=args.input_size)
    model = LitModel(num_classes=args.num_classes, transfer=args.is_transfer, learning_rate=args.learning_rate, resnet_scale=args.resnet_scale)
    if args.is_distributed:
        trainer = pl.Trainer(logger=logger, max_epochs=args.max_epochs, accelerator="gpu",callbacks=[checkpoint_callback],strategy="ddp")
    else:
        trainer = pl.Trainer(logger=logger, max_epochs=args.max_epochs, accelerator="gpu",callbacks=[checkpoint_callback])
    trainer.fit(model, dm)
    print("end....")

if __name__ == "__main__":
    main()
