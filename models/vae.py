# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class VAEncoder(nn.Module):
#     def __init__(self, in_channels, latent_dim):
#         super(VAEncoder, self).__init__()
        
#         # 编码器网络层（卷积层 + 扁平化层）
#         self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
#         # 平均和对数方差（用于计算潜在变量的分布）
#         self.fc1 = nn.Linear(128 * 8 * 8, 256)
#         self.fc_mean = nn.Linear(256, latent_dim)  # 均值
#         self.fc_log_var = nn.Linear(256, latent_dim)  # 对数方差
        
#     def forward(self, x):
#         # 卷积 + 激活函数
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
        
#         # 扁平化
#         x = x.view(x.size(0), -1)
        
#         # 通过全连接层
#         x = F.relu(self.fc1(x))
        
#         # 计算均值和对数方差
#         mean = self.fc_mean(x)
#         log_var = self.fc_log_var(x)
        
#         return mean, log_var
    
# class VADecoder(nn.Module):
#     def __init__(self, out_channels, latent_dim):
#         super(VADecoder, self).__init__()
        
#         # 全连接层从潜在空间到图像空间
#         self.fc1 = nn.Linear(latent_dim, 256)
#         self.fc2 = nn.Linear(256, 128 * 8 * 8)
        
#         # 解码器的卷积层
#         self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.deconv3 = nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        
#     def forward(self, z):
#         # 从潜在变量到全连接
#         x = F.relu(self.fc1(z))
#         x = F.relu(self.fc2(x))
        
#         # 重塑成卷积输入
#         x = x.view(x.size(0), 128, 8, 8)
        
#         # 反卷积解码
#         x = F.relu(self.deconv1(x))
#         x = F.relu(self.deconv2(x))
#         x = self.deconv3(x)
        
#         return x
    
# class RVAE(nn.Module):
#     def __init__(self, in_channels, out_channels, latent_dim):
#         super(RVAE, self).__init__()
#         self.encoder = VAEncoder(in_channels, latent_dim)
#         self.decoder = VADecoder(out_channels, latent_dim)

#     def reparameterize(self, mean, log_var):
#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)
#         return mean + eps * std

#     def forward(self, x):
#         mean, log_var = self.encoder(x)
#         z = self.reparameterize(mean, log_var)
#         reconstructed_x = self.decoder(z)
#         return reconstructed_x, mean, log_var
#     def get_loss(self,x,r=0.5):
#         reconstructed_x, mean, log_var = self.forward(x)
#         reconstruction_loss = F.mse_loss(reconstructed_x, x, reduction='sum')
#         kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
#         return r*reconstruction_loss + kl_loss


import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim,dim_dot=1):
        super(VAEncoder, self).__init__()
        
        # 编码器的全连接层
        self.fc1 = nn.Linear(input_dim, 512)  # 输入维度是 batch, dim
        self.fc2 = nn.Linear(512, 512)
        
        # 平均和对数方差（用于计算潜在变量的分布）
        self.fc_mean = nn.Linear(512, latent_dim)  # 均值
        self.fc_log_var = nn.Linear(512, latent_dim)  # 对数方差
        
    def forward(self, x):
        # 通过全连接层 + 激活函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # 计算均值和对数方差
        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        
        return mean, log_var
    

class VADecoder(nn.Module):
    def __init__(self, output_dim, latent_dim,dim_dot=1):
        super(VADecoder, self).__init__()
        
        # 解码器的全连接层
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, output_dim)  # 输出维度是 dim

    def forward(self, z):
        # 通过全连接层 + 激活函数
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class RVAE(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim,dim_dot=1):
        super(RVAE, self).__init__()
        self.encoder = VAEncoder(input_dim, latent_dim,dim_dot)
        self.decoder = VADecoder(output_dim, latent_dim,dim_dot)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mean, log_var

    def get_rec_loss(self, x, r=0.5):
        reconstructed_x, mean, log_var = self.forward(x)
        reconstruction_loss = F.mse_loss(reconstructed_x, x, reduction='mean')
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reconstructed_x,r * reconstruction_loss + kl_loss