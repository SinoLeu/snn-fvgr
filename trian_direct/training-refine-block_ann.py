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
# from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from utils.pl_data_loader import StanfordCarsDataRefineModule
from models.refine_module import FeatureRefine


def remove_prefix_from_state_dict(state_dict, prefix="feature_extractor."):
    """
    从 state_dict 中删除指定前缀
    """
    return {key.replace(prefix, ""): value for key, value in state_dict.items()}


def freeze_model(model: nn.Module):
    """
    冻结模型的所有参数，即设置 requires_grad=False
    
    Args:
        model (nn.Module): 要冻结的 PyTorch 模型
    
    Returns:
        nn.Module: 冻结后的模型
    """
    for param in model.parameters():
        param.requires_grad = False
    return model

class LitModel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=0.1, transfer=True,resnet_scale:str = '50',load_checkpoints_path:str="./logs/fine_tune_resnet50/version_8/checkpoints/best_model.ckpt",train_component:str='att'):
        super().__init__()
        
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        
        ## 
        channel_dot = 1
        if resnet_scale == '18':
            if transfer:
                self.feature_extractor = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.feature_extractor = models.resnet18(weights=None)
        elif resnet_scale == '50':
            channel_dot = 4
            if transfer:
                self.feature_extractor = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                self.feature_extractor = models.resnet50(weights=None)
        elif resnet_scale == '101':
            channel_dot = 8
            if transfer:
                self.feature_extractor = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            else:
                self.feature_extractor = models.resnet101(weights=None)
        elif resnet_scale == '34':
            channel_dot = 2
            if transfer:
                self.feature_extractor = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            else:
                self.feature_extractor = models.resnet34(weights=None)
        best_model = torch.load(load_checkpoints_path)
        # 应用到你的 best_model['state_dict']
        in_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(in_features, num_classes)
        
        best_model['state_dict'] = remove_prefix_from_state_dict(best_model['state_dict'])
        self.feature_extractor.load_state_dict(best_model['state_dict'])
        # self.feature_extractor.fc = nn.Identity()
        self.feature_extractor.eval()
        freeze_model(self.feature_extractor)
        self.refine_block = FeatureRefine(class_nums=num_classes,channel_dot=channel_dot)
        self.classifier = nn.Identity()
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass",num_classes=self.num_classes)
        
        # if train_component == 'att':
        #     self.refine_block.rvae.eval()
        # elif train_component == 'vae':
        #     self.refine_block.trilli_att_module.eval()
        # self.train_component = train_component
    # returns the size of the output tensor going into the Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(tmp_input) 
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
        
    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        x,mid_out = self.feature_extractor(x,return_mid=True)

        return x,mid_out
    
    # will be used during inference
    def forward(self, x):
       x,mid_out = self._forward_features(x)
       return x,mid_out
    
    def training_step(self, batch):
        batch, gt = batch[0], batch[1]
        out,mid_out = self.forward(batch)
        feature_maps = mid_out[:4]
        out = mid_out[-1]
        y1,y2,vae_loss = self.refine_block(feature_maps,out)
        current_epoch = self.current_epoch
        loss = self.refine_block.get_fr_loss(gt,y1,y2,vae_loss,current_epoch)
        # loss = self.criterion(out, gt)
        self.log("train/loss", loss)
        acc_y1 = self.accuracy(y1, gt)
        self.log("train/y1_acc", acc_y1)
        acc_y2 = self.accuracy(y2, gt)
        self.log("train/y2_acc", acc_y2)
        # if self.train_component == 'att':
        #     acc_y1 = self.accuracy(y1, gt)
        #     self.log("train/y1_acc", acc_y1)
        # else:
        #     acc_y2 = self.accuracy(y2, gt)
        #     self.log("train/y2_acc", acc_y2)

        return loss
    
    def validation_step(self, batch, batch_idx):
        batch, gt = batch[0], batch[1]
        # out = self.feature_extractor(batch)
        out,mid_out = self.forward(batch)
        feature_maps = mid_out[:4]
        out = mid_out[-1]
        y1,y2,vae_loss = self.refine_block(feature_maps,out)
        current_epoch = self.current_epoch
        loss = self.refine_block.get_fr_loss(gt,y1,y2,vae_loss,current_epoch)
        
        self.log("val/loss", loss)
        acc_y1 = self.accuracy(y1, gt)
        acc_y2 = self.accuracy(y2, gt)
        self.log("val/acc", acc_y1)
        self.log("val/acc", acc_y2)
        # if self.train_component == 'att':
        #     acc_y1 = self.accuracy(y1, gt)
        #     self.log("val/acc", acc_y1)
        # elif self.train_component == 'vae':
        #     acc_y2 = self.accuracy(y2, gt)
        #     self.log("val/acc", acc_y2)
        self.log("val/vae_loss", vae_loss)
        # loss = self.criterion(out, gt)
        
        # acc = self.accuracy(out, gt)
        # self.log("val/acc", acc)
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
        
        
# # 设置
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
    parser.add_argument('--load_checkpoints_path', type=str, default="./logs/fine_tune_resnet50/version_8/checkpoints/best_model.ckpt", help="load trained cpkt")
    # parser.add_argument('--train_component', type=str, default="att", help="trained component")
    return parser.parse_args()


def main():
    args = parse_args()
    logger_name = f"refine_resnet{args.resnet_scale}"
    logger = CSVLogger(args.checkpoint_dir, name=logger_name)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/acc",           # 监控验证集准确率
        mode="max",                  # 追踪最大值
        save_top_k=1,                # 保存最佳模型
        verbose=True,                # 输出日志
        filename="best_model"        # 文件名
    )

    dm = StanfordCarsDataRefineModule(batch_size=args.batch_size, train_dir=args.train_dir, test_dir=args.test_dir, input_size=args.input_size)
    model = LitModel(num_classes=args.num_classes, transfer=args.is_transfer, learning_rate=args.learning_rate, resnet_scale=args.resnet_scale,train_component=None)
    if args.is_distributed:
        trainer = pl.Trainer(logger=logger, max_epochs=args.max_epochs, accelerator="gpu",callbacks=[checkpoint_callback],strategy="ddp")
    else:
        trainer = pl.Trainer(logger=logger, max_epochs=args.max_epochs, accelerator="gpu",callbacks=[checkpoint_callback])
    trainer.fit(model, dm)
    print("end....")

if __name__ == "__main__":
    main()

## 
## python training-refine-block_ann.py --batch_size 32 --learning_rate 1e-3 --input_size 300 --train_dir '../data/cars/train' --test_dir '../data/cars/test' --resnet_scale '50' --max_epochs 200  --checkpoint_dir 'logs' --num_classes 196 --is_distributed  --is_transfer --load_checkpoints_path "./logs/fine_tune_resnet50/version_8/checkpoints/best_model.ckpt" 
##--train_component 'att'

## python training-refine-block_ann.py --batch_size 32 --learning_rate 1e-3 --input_size 300 --train_dir '../data/cars/train' --test_dir '../data/cars/test' --resnet_scale '50' --max_epochs 200  --checkpoint_dir 'logs' --num_classes 196 --is_distributed  --is_transfer --load_checkpoints_path "./logs/fine_tune_resnet50/version_8/checkpoints/best_model.ckpt" --train_component 'att'
