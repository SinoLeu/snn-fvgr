import torch
import torch.nn as nn
import torch.nn.functional as F

from models.trilli_att_module import TrilliAttModules
from models.vae import RVAE


class FeatureRefine(nn.Module):
    def __init__(self,input_dim=512,output_dim=512,latent_dim=256,
                   target_height=12, target_width=12,mid_channel=256,
                   input_layers=4,compression_out_channels=512,class_nums=100,channel_dot=1):
        super().__init__()
        input_dim = input_dim * channel_dot
        output_dim = output_dim * channel_dot
        latent_dim = latent_dim * channel_dot
        
        self.trilli_att_module = TrilliAttModules(target_height=target_height, target_width=target_width, mid_channel=mid_channel, input_layers=input_layers, compression_out_channels=compression_out_channels, compression_target_size=1,channel_dot=channel_dot)
        self.rvae = RVAE(input_dim=input_dim,output_dim=output_dim,latent_dim=latent_dim)
        self.head = nn.Linear(output_dim,class_nums)
    def forward(self,feature_maps,out):
        
        output = self.trilli_att_module(feature_maps)
        # print(output.shape)
        reconstructed_x,vae_loss = self.rvae.get_rec_loss(out,r=0.5)
        # print(reconstructed_x.shape)
        # print(torch.flatten(output, 1).shape)
        y1 = self.head(torch.flatten(output, 1))
        y2 = self.head(torch.flatten(reconstructed_x, 1))
        return y1,y2,vae_loss
    def get_fr_loss(self,y,y1,y2,vae_loss):
        # y1,y2,vae_loss = self.forward(feature_maps,out)
        criterion = nn.CrossEntropyLoss()
        return criterion(y1,y) + criterion(y2,y) +  vae_loss
        ##  self.criterion(out, gt)
