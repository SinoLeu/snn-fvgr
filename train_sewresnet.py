# import os
# import torch
# from torch import nn, optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, models, transforms
# # import pytorch_lightning as pl
# # from pytorch_lightning import Trainer
# from torchvision import models, transforms
# import torch
# # from timm import create_model
# from torchvision import transforms, datasets
# import lightning as L
# import timm
# # import os
# from torch.optim.lr_scheduler import StepLR

# import pytorch_lightning as pl
# # your favorite machine learning tracking tool
# from pytorch_lightning.loggers import WandbLogger
# import argparse
# import torch
# from torch import nn
# from torch.nn import functional as F
# from torch.utils.data import random_split, DataLoader
# from torch.optim.lr_scheduler import CosineAnnealingLR,OneCycleLR
# from torchmetrics import Accuracy

# from torchvision import transforms
# # from torchvision.datasets import StanfordCars
# # from torchvision.datasets.utils import download_url
# import torchvision.models as models
# # from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.loggers import CSVLogger
# from utils.pl_data_loader import StanfordCarsDataModule
# from models.ml_decoder import add_ml_decoder_head,add_linear_head
# from losses import ASLSingleLabel
# ## python fine-tine-resnet_ann.py --batch_size 64 --learning_rate 0.1 --input_size 300 --train_dir '../data/cars/train' --test_dir '../data/cars/test' --resnet_scale '50' --max_epochs 150  --checkpoint_dir 'logs' --num_classes 196 --is_distributed  --is_transfer
# from torchvision.models.feature_extraction import get_graph_node_names
# from models.s_model import get_sewresnet
# from spikingjelly.activation_based import neuron, functional, surrogate, layer
# ## fine-tune resnet
# class LitModel(pl.LightningModule):
#     def __init__(self, num_classes, learning_rate=0.1, transfer=True,resnet_scale:str = '50',Timestep=4,weight_decay=1e-5):
#         super().__init__()
        
#         self.save_hyperparameters()
#         self.learning_rate = learning_rate
#         self.num_classes = num_classes
#         self.weight_decay = weight_decay
#         ## from torchvision.models.feature_extraction import get_graph_node_names
#         # self.feature_extractor = timm.create_model(f'resnet{resnet_scale}.a1_in1k', pretrained=transfer, num_classes=num_classes)
#         self.feature_extractor = get_sewresnet(arch=resnet_scale,num_classes=num_classes,T=Timestep,connect_f='ADD')
#         # print(get_graph_node_names(self.feature_extractor))
#         self.criterion = nn.CrossEntropyLoss()
#         self.accuracy = Accuracy(task="multiclass",num_classes=self.num_classes)
        
#     # returns the feature tensor from the conv block
#     def _forward_features(self, x):
#         x = self.feature_extractor(x)
#         functional.reset_net(self.feature_extractor)
#         return x
    
#     # will be used during inference
#     def forward(self, x):
#        x = self._forward_features(x)       
#        return x
    
#     def training_step(self, batch):
#         batch, gt = batch[0], batch[1]
#         out = self.forward(batch)
#         loss = self.criterion(out, gt)

#         acc = self.accuracy(out, gt)

#         self.log("train/loss", loss)
#         self.log("train/acc", acc)

#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         batch, gt = batch[0], batch[1]
#         out = self.forward(batch)
#         loss = self.criterion(out, gt)

#         self.log("val/loss", loss)

#         acc = self.accuracy(out, gt)
#         self.log("val/acc", acc)

#         return loss
    
#     def test_step(self, batch, batch_idx):
#         batch, gt = batch[0], batch[1]
#         out = self.forward(batch)
#         loss = self.criterion(out, gt)
        
#         return {"loss": loss, "outputs": out, "gt": gt}
    
#     def test_epoch_end(self, outputs):
#         loss = torch.stack([x['loss'] for x in outputs]).mean()
#         output = torch.cat([x['outputs'] for x in outputs], dim=0)
        
#         gts = torch.cat([x['gt'] for x in outputs], dim=0)
        
#         self.log("test/loss", loss)
#         acc = self.accuracy(output, gts)
#         self.log("test/acc", acc)
        
#         self.test_gts = gts
#         self.test_output = output
    
#     def configure_optimizers(self):
#         # return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
#         optimizer = optim.SGD(self.parameters(), lr=self.learning_rate,weight_decay=self.weight_decay)
#         # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
#         scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=0)
#         # steps_per_epoch = self.train_loader_len
#         # scheduler = OneCycleLR(optimizer, max_lr=self.learning_rate, steps_per_epoch=steps_per_epoch, epochs=self.trainer.max_epochs,
#         #                                 pct_start=0.2)
#         return {
#             'optimizer': optimizer,
#             'lr_scheduler': scheduler,
#             'monitor': 'val_loss'
#         }

# # # 设置 TensorBoardLogger
# def parse_args():
#     from config.config import parse_args_yml
#     args = parse_args_yml('config/sew_train_resnet.yml')
#     return args


# def main():
#     args = parse_args()
#     logger_name = f"{args.env_train_name}_{args.resnet_scale}"
#     logger = CSVLogger(args.checkpoint_dir, name=logger_name)

#     checkpoint_callback = ModelCheckpoint(
#         monitor="val/acc",           # 监控验证集准确率
#         mode="max",                  # 追踪最大值
#         save_top_k=1,                # 保存最佳模型
#         verbose=True,                # 输出日志
#         filename="best_model"        # 文件名
#     )

#     dm = StanfordCarsDataModule(batch_size=args.batch_size, train_dir=args.train_dir, test_dir=args.test_dir, input_size=args.input_size)
#     model = LitModel(num_classes=args.num_classes, transfer=args.is_transfer, learning_rate=args.learning_rate, resnet_scale=args.resnet_scale,Timestep=args.time_step,weight_decay=args.weight_decay)
#     if args.is_distributed:
#         trainer = pl.Trainer(logger=logger, max_epochs=args.max_epochs, accelerator="gpu",callbacks=[checkpoint_callback],strategy="ddp")
#     else:
#         trainer = pl.Trainer(logger=logger, max_epochs=args.max_epochs, accelerator="gpu",callbacks=[checkpoint_callback])
#     trainer.fit(model, dm)
#     print("end....")

# if __name__ == "__main__":
#     main()

# ## python train_sewresnet.py  >> ./output.log 2>&1
# ## python train_sewresnet.py --batch_size 64 --learning_rate 0.1 --input_size 300 --train_dir '../data/cars/train' --test_dir '../data/cars/test' --resnet_scale '50' --max_epochs 100  --checkpoint_dir 'logs' --num_classes 196 --is_distributed  --is_transfer
# ## python train_sewresnet.py --batch_size 64 --learning_rate 0.1 --input_size 300 --train_dir '../data/cars/train' --test_dir '../data/cars/test' --resnet_scale '101' --max_epochs 100  --checkpoint_dir 'logs' --num_classes 196 --is_distributed  --is_transfer  >> ./output.log 2>&1
# ## python train_sewresnet.py --batch_size 64 --learning_rate 0.1 --input_size 300 --train_dir '../data/cars/train' --test_dir '../data/cars/test' --resnet_scale '50' --max_epochs 200  --checkpoint_dir 'logs' --num_classes 196 --is_distributed  --is_transfer
# ## python train_sewresnet.py --batch_size 64 --learning_rate 0.1 --input_size 300 --train_dir '../data/cars/train' --test_dir '../data/cars/test' --resnet_scale '50' --max_epochs 200  --checkpoint_dir 'logs' --num_classes 196 --is_distributed  --is_transfer
