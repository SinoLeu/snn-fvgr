import torch
import torch.nn as nn
import torch.nn.functional as F

from trilli_att_module import TrilliAttModules
from vae import RVAE


class FeatureRefine(nn.Module):
    def __init__(self,input_dim=512,output_dim=512,latent_dim=256,
                   target_height=52, target_width=52,mid_channel=256,
                   input_layers=4,compression_out_channels=512,class_nums=100):
          
        self.trilli_att_module = TrilliAttModules(target_height=target_height, target_width=target_width, mid_channel=mid_channel, input_layers=input_layers, compression_out_channels=compression_out_channels, compression_target_size=1)
        self.rvae = RVAE(input_dim=input_dim,output_dim=output_dim,latent_dim=latent_dim)
        self.head = nn.Linear(512,class_nums)
    def forward(self,feature_maps,out):
        output = self.tr_att_module(feature_maps)
        reconstructed_x,vae_loss = self.rvae.get_rec_loss(out,r=0.5)
        y1 = self.head(output)
        y2 = self.head(reconstructed_x)
        return y1,y2,vae_loss
    def get_fr_loss(self,feature_maps,out,y):
        y1,y2,vae_loss = self.forward(feature_maps,out)
        critien = nn.CrossEntropyLoss()
        return critien(y,y1) + critien(y,y2) + vae_loss
