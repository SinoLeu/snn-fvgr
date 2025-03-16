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

from torchvision.models.feature_extraction import create_feature_extractor
import timm
from models.pim_module import FPN, WeaklySelector, GCNCombiner,SharedPluginMoodel


def add_pim_module(backbone,return_nodes,img_size,num_classes,num_selects,fpn_size,comb_proj_size,shared_fpn,shared_selector,shared_combiner):
    return SharedPluginMoodel(
            backbone=backbone,
            return_nodes=return_nodes,
            img_size=img_size,
            use_fpn=True,
            fpn_size=fpn_size,
            proj_type="Conv",
            upsample_type="Bilinear",
            use_selection=True,
            num_classes=num_classes,
            num_selects=num_selects,
            use_combiner=True,
            comb_proj_size=comb_proj_size,
            fpn=shared_fpn,          # 传入共享的 FPN
            selector=shared_selector, # 传入共享的 Selector
            combiner=shared_combiner  # 传入共享的 Combiner
        )

## fine-tune resnet
# returns the size of the output tensor going into the Linear layer from the conv block.
class LitModel(pl.LightningModule):
    def __init__(self, num_classes, 
                 learning_rate=0.1, 
                 transfer=True,
                 resnet_scale:str = '50',
                 return_nodes={},
                 img_size=224,num_selects={},
                 fpn_size=256,comb_proj_size=512,
                 loss_param = {
                    'lambda_s':0,'lambda_n':0,'lambda_b':0,'lambda_c':0
                 },update_freq=2,weight_decay=1e-3
                ):
        super().__init__()
        
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        ## 
        backbone = timm.create_model(f'resnet{resnet_scale}.a1_in1k', pretrained=transfer, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass",num_classes=self.num_classes)
        
        backbone1 = create_feature_extractor(backbone, return_nodes=return_nodes)
        rand_in = torch.randn(1, 3, img_size, img_size)
        outs = backbone1(rand_in)
        self.update_freq =  update_freq
        # print(outs.shape)
        # 创建共享的插件模块
        shared_fpn = FPN(
            outs,
            fpn_size, proj_type="Conv", upsample_type="Bilinear"
        )
        shared_selector = WeaklySelector(
            outs,
            num_classes, num_selects, fpn_size
        )
        shared_combiner = GCNCombiner(
            total_num_selects=sum(num_selects.values()),
            num_classes=num_classes,
            fpn_size=fpn_size
        )

        self.feature_extractor = add_pim_module(
            backbone=backbone,return_nodes=return_nodes,
            img_size=img_size,num_classes=num_classes,
            num_selects=num_selects,
            fpn_size=fpn_size,comb_proj_size=comb_proj_size,
            shared_fpn=shared_fpn,shared_selector=shared_selector,shared_combiner=shared_combiner
        )
        
        self.lambda_s = loss_param['lambda_s']
        self.lambda_n = loss_param['lambda_n']
        self.lambda_b = loss_param['lambda_b']
        self.lambda_c = loss_param['lambda_c']
        self.num_classes = num_classes
        self.weight_decay = weight_decay
    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        x = self.feature_extractor(x)
        return x
    
    # will be used during inference
    def forward(self, x):
       x = self._forward_features(x)       
       return x
    
    def compute_loss(self,outs,labels):
        loss = 0.
        batch_size = labels.size(0)
        for name in outs:
            if "select_" in name:
                if self.lambda_s != 0:
                    S = outs[name].size(1)
                    logit = outs[name].view(-1, self.num_classes).contiguous()
                    loss_s = nn.CrossEntropyLoss()(logit, 
                                                       labels.unsqueeze(1).repeat(1, S).flatten(0))
                    loss += self.lambda_s * loss_s
                else:
                    loss_s = 0.
            
            elif "drop_" in name:
                if self.lambda_n != 0:
                    S = outs[name].size(1)
                    logit = outs[name].view(-1, self.num_classes).contiguous()
                    n_preds = nn.Tanh()(logit)
                    labels_0 = torch.zeros([batch_size * S, self.num_classes]) - 1
                    labels_0 = labels_0.to(self.device)
                    loss_n = nn.MSELoss()(n_preds, labels_0)
                    loss += self.lambda_n * loss_n
                else:
                    loss_n = 0.0

            elif "layer" in name:
                if self.lambda_b != 0:
                    ### here using 'layer1'~'layer4' is default setting, you can change to your own
                    loss_b = nn.CrossEntropyLoss()(outs[name].mean(1), labels)
                    loss += self.lambda_b * loss_b
                else:
                    loss_b = 0.0
            elif "comb_outs" in name:
                if self.lambda_c != 0:
                    loss_c = nn.CrossEntropyLoss()(outs[name], labels)
                    loss += self.lambda_c * loss_c
            elif "ori_out" in name:
                    loss_ori = F.cross_entropy(outs[name], labels)
                    loss += loss_ori
            else:
                print(name)
        # print(loss)
        return loss / self.update_freq

    # def get_top5_out(self,out):
    #     pass
    
    def get_top5_out(self, outs):
        # Step 1: 统一张量形状并拼接
        result = torch.cat(
            [outs[name].unsqueeze(1) if outs[name].dim() == 2 else outs[name] for name in outs],
            dim=1
        )  # 形状 (batch_size, dim, num_classes)，例如 (32, 14, 10)

        # Step 2: 对每个样本的 dim 维度取 Top-5
        # result 形状 (batch_size, dim, num_classes)
        batch_size, dim, num_classes = result.shape
        
        # 沿着 dim 维度排序，取 Top-5
        values, indices = torch.topk(result, k=5, dim=1, largest=True)  # values 形状 (batch_size, 5, num_classes)

        # Step 3: 对 Top-5 值取均值
        final_logit = values.mean(dim=1)  # 形状 (batch_size, num_classes)，例如 (32, 10)

        return final_logit
    def training_step(self, batch):
        batch, gt = batch[0], batch[1]
        # print(batch.shape)
        outs = self.forward(batch)
        loss = self.compute_loss(outs,gt)
        out = self.get_top5_out(outs)
        # out = outs["comb_outs"]
        
        # loss = self.criterion(out, gt)

        acc = self.accuracy(out, gt)

        self.log("train/loss", loss)
        self.log("train/acc", acc)

        return loss
    
    def validation_step(self, batch, batch_idx):
        batch, gt = batch[0], batch[1]
        outs = self.forward(batch)
        loss = self.compute_loss(outs,gt)
        
        self.log("val/loss", loss)
        out = self.get_top5_out(outs)
        acc = self.accuracy(out, gt)
        self.log("val/acc", acc)

        return loss
    
    def test_step(self, batch, batch_idx):
        batch, gt = batch[0], batch[1]
        outs = self.forward(batch)
        loss = self.compute_loss(outs,gt)
        out = self.get_top5_out(outs)
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
        optimizer = optim.SGD(self.feature_extractor.parameters(), lr=self.learning_rate,weight_decay=self.weight_decay)
        # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=0)
        # steps_per_epoch = self.train_loader_len
        # scheduler = OneCycleLR(optimizer, max_lr=self.learning_rate, steps_per_epoch=steps_per_epoch, epochs=self.trainer.max_epochs,
        #                                 pct_start=0.2)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

# # 设置 TensorBoardLogger
def parse_args():
    from config.config import parse_args_yml
    args = parse_args_yml('config/pim_train_resnet.yml')
    return args


def main():
    args = parse_args()
    logger_name = f"pim_fine_tune_resnet{args.resnet_scale}"
    logger = CSVLogger(args.checkpoint_dir, name=logger_name)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/acc",           # 监控验证集准确率
        mode="max",                  # 追踪最大值
        save_top_k=1,                # 保存最佳模型
        verbose=True,                # 输出日志
        filename="best_model"        # 文件名
    )

    dm = StanfordCarsDataModule(batch_size=args.batch_size, train_dir=args.train_dir, test_dir=args.test_dir, input_size=args.img_size)
    model = LitModel(num_classes=args.num_classes, transfer=args.is_transfer, learning_rate=args.learning_rate,
                     resnet_scale=args.resnet_scale,
                     return_nodes=args.return_nodes,
                     img_size=args.img_size,num_selects=args.num_selects,
                     fpn_size=args.fpn_size,comb_proj_size=args.comb_proj_size,
                     loss_param = args.loss_param,update_freq=args.update_freq,
                     weight_decay=args.weight_decay)
    if args.is_distributed:
        trainer = pl.Trainer(logger=logger, max_epochs=args.max_epochs, accelerator="gpu",callbacks=[checkpoint_callback],strategy="ddp",precision="16-mixed")
    else:
        trainer = pl.Trainer(logger=logger, max_epochs=args.max_epochs, accelerator="gpu",callbacks=[checkpoint_callback],precision="16-mixed")
    trainer.fit(model, dm)
    print("end....")

if __name__ == "__main__":
    main()


## python fine_tine_resnet_ml_decoder.py --batch_size 64 --learning_rate 0.1 --input_size 300 --train_dir '../data/cars/train' --test_dir '../data/cars/test' --resnet_scale '101' --max_epochs 100  --checkpoint_dir 'logs' --num_classes 196 --is_distributed  --is_transfer  >> ./output.log 2>&1
## python fine_tine_resnet_ml_decoder.py --batch_size 64 --learning_rate 0.1 --input_size 300 --train_dir '../data/cars/train' --test_dir '../data/cars/test' --resnet_scale '50' --max_epochs 200  --checkpoint_dir 'logs' --num_classes 196 --is_distributed  --is_transfer
## python fine_tine_resnet_ml_decoder.py --batch_size 64 --learning_rate 0.1 --input_size 300 --train_dir '../data/cars/train' --test_dir '../data/cars/test' --resnet_scale '50' --max_epochs 200  --checkpoint_dir 'logs' --num_classes 196 --is_distributed  --is_transfer
## python fine-tine-resnet_ann.py --batch_size 64 --learning_rate 0.1 --input_size 300 --train_dir '../data/cars/train' --test_dir '../data/cars/test' --resnet_scale '50' --max_epochs 150  --checkpoint_dir 'logs' --num_classes 196 --is_distributed  --is_transfer
