import torch
import torch.nn as nn
import torch.nn.functional as F

from models.trilli_att_module import TrilliAttModules
from models.vae import RVAE


class FeatureRefine_Att(nn.Module):
    def __init__(self,input_dim=512,output_dim=512,latent_dim=256,
                   target_height=12, target_width=12,mid_channel=256,
                   input_layers=4,compression_out_channels=512,class_nums=100,channel_dot=1):
        super().__init__()
        input_dim = input_dim * channel_dot
        output_dim = output_dim * channel_dot
        latent_dim = latent_dim * channel_dot        

        self.trilli_att_module = TrilliAttModules(target_height=target_height, target_width=target_width, mid_channel=mid_channel, 
                                                  input_layers=input_layers, compression_out_channels=compression_out_channels, 
                                                  compression_target_size=1,channel_dot=channel_dot)
        self.dropout = nn.Dropout(0.3)
        self.head1 = nn.Linear(output_dim,class_nums)
        
    def forward(self,feature_maps):
        output = self.trilli_att_module(feature_maps)
        y1 = self.head1(self.dropout(torch.flatten(output, 1)))
        
        return y1
    def get_att_loss(self,y,y1):
        # y1,y2,vae_loss = self.forward(feature_maps,out)
        criterion = nn.CrossEntropyLoss()
        return criterion(y1,y)

class FeatureRefine_RVAE():
    def __init__(self,input_dim=512,output_dim=512,latent_dim=256,class_nums=100,channel_dot=1):
        super().__init__()
        input_dim = input_dim * channel_dot
        output_dim = output_dim * channel_dot
        latent_dim = latent_dim * channel_dot
        
        self.rvae = RVAE(input_dim=input_dim,output_dim=output_dim,latent_dim=latent_dim)
        self.dropout = nn.Dropout(0.3)
        self.head2 = nn.Linear(output_dim,class_nums)
    def forward(self,out):
        reconstructed_x,vae_loss = self.rvae.get_rec_loss(out,r=0.8)
        y2 = self.head2(self.dropout(torch.flatten(reconstructed_x, 1)))
        return y2,vae_loss
    
    def get_rvae_loss(self,y,y2,vae_loss):
        criterion = nn.CrossEntropyLoss()
        return criterion(y2,y) + vae_loss
        
class FeatureRefine(nn.Module):
    def __init__(self,input_dim=512,output_dim=512,latent_dim=256,
                   target_height=12, target_width=12,mid_channel=256,
                   input_layers=4,compression_out_channels=512,class_nums=100,channel_dot=1):
        super().__init__()
        input_dim = input_dim * channel_dot
        output_dim = output_dim * channel_dot
        latent_dim = latent_dim * channel_dot
        
        self.trilli_att_module = TrilliAttModules(target_height=target_height, target_width=target_width, mid_channel=mid_channel, 
                                                  input_layers=input_layers, compression_out_channels=compression_out_channels, 
                                                  compression_target_size=1,channel_dot=channel_dot)
        self.rvae = RVAE(input_dim=input_dim,output_dim=output_dim,latent_dim=latent_dim)
        self.dropout = nn.Dropout(0.3)
        self.head1 = nn.Linear(output_dim,class_nums)
        self.head2 = nn.Linear(output_dim,class_nums)
    def forward(self,feature_maps,out):
        ## `dropo`
        output = self.trilli_att_module(feature_maps)
        reconstructed_x,vae_loss = self.rvae.get_rec_loss(out,r=0.8)
        # print(output.shape)
        # print(reconstructed_x.shape)
        # print(torch.flatten(output, 1).shape)
        y1 = self.head1(self.dropout(torch.flatten(output, 1)))
        y2 = self.head2(self.dropout(torch.flatten(reconstructed_x, 1)))
        return y1,y2,vae_loss
    def get_att_loss(self,y,y1):
        # y1,y2,vae_loss = self.forward(feature_maps,out)
        criterion = nn.CrossEntropyLoss()
        return criterion(y1,y)
    def get_rvae_loss(self,y,y2,vae_loss):
        criterion = nn.CrossEntropyLoss()
        return criterion(y2,y) + vae_loss
        # return criterion(y1,y)
