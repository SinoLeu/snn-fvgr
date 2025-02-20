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
from torch.optim.lr_scheduler import CosineAnnealingLR,OneCycleLR
from torchmetrics import Accuracy

from torchvision import transforms
# from torchvision.datasets import StanfordCars
# from torchvision.datasets.utils import download_url
import torchvision.models as models
# from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from utils.pl_data_loader import StanfordCarsDataModule
from models.ml_decoder import add_ml_decoder_head
from losses import ASLSingleLabel
## python fine-tine-resnet_ann.py --batch_size 64 --learning_rate 0.1 --input_size 300 --train_dir '../data/cars/train' --test_dir '../data/cars/test' --resnet_scale '50' --max_epochs 150  --checkpoint_dir 'logs' --num_classes 196 --is_distributed  --is_transfer

## fine-tune resnet
class LitModel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=0.1, transfer=True,resnet_scale:str = '50',decoder_embedding=768,num_of_groups=-1,train_loader_len = 100):
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
        self.train_loader_len = train_loader_len
        # in_features = self.feature_extractor.fc.in_features
        # self.feature_extractor.fc = nn.Linear(in_features, num_classes)
        # self.classifier = nn.Identity()
        self.criterion = ASLSingleLabel()
        self.accuracy = Accuracy(task="multiclass",num_classes=self.num_classes)
        self.feature_extractor = add_ml_decoder_head(self.feature_extractor,num_classes=num_classes,num_of_groups=num_of_groups,decoder_embedding=decoder_embedding)
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
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        # scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=0)
        steps_per_epoch = self.train_loader_len
        scheduler = OneCycleLR(optimizer, max_lr=self.learning_rate, steps_per_epoch=steps_per_epoch, epochs=self.trainer.max_epochs,
                                        pct_start=0.2)
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

    dm = StanfordCarsDataModule(batch_size=args.batch_size, train_dir=args.train_dir, test_dir=args.test_dir, input_size=args.input_size)
    model = LitModel(num_classes=args.num_classes, transfer=args.is_transfer, learning_rate=args.learning_rate, resnet_scale=args.resnet_scale)
    if args.is_distributed:
        trainer = pl.Trainer(logger=logger, max_epochs=args.max_epochs, accelerator="gpu",callbacks=[checkpoint_callback],strategy="ddp",train_loader_len=len(dm.train_dataloader()))
    else:
        trainer = pl.Trainer(logger=logger, max_epochs=args.max_epochs, accelerator="gpu",callbacks=[checkpoint_callback],train_loader_len=len(dm.train_dataloader()))
    trainer.fit(model, dm)
    print("end....")

if __name__ == "__main__":
    main()


## python fine_tine_resnet_ml_decoder.py --batch_size 64 --learning_rate 0.1 --input_size 300 --train_dir '../data/cars/train' --test_dir '../data/cars/test' --resnet_scale '50' --max_epochs 200  --checkpoint_dir 'logs' --num_classes 196 --is_distributed  --is_transfer

## python fine_tine_resnet_ml_decoder.py --batch_size 64 --learning_rate 0.3 --input_size 300 --train_dir '../data/cars/train' --test_dir '../data/cars/test' --resnet_scale '50' --max_epochs 200  --checkpoint_dir 'logs' --num_classes 196 --is_distributed  --is_transfer