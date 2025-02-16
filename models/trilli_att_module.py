import torch
import torch.nn as nn
import torch.nn.functional as F

class TrilinearAttention(nn.Module):
    def __init__(self, target_height, target_width, mid_channel,input_layers=4):
        super(TrilinearAttention, self).__init__()
        self.target_height = target_height
        self.target_width = target_width
        self.mid_channel = mid_channel  # 目标降维的通道数
        in_channel = [((2)**i)*64 for i in range(input_layers)]
        # print(in_channel)
        # 1x1 卷积层用于降维
        self.channel_reduction = nn.Conv2d(in_channels=sum(in_channel), out_channels=self.mid_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, feature_maps):
        """
        feature_maps: 输入的多个特征图列表，每个特征图的形状为 [batch, channels, height, width]
        输出: 经过三线性注意力调整后的融合特征图，形状为 [batch, mid_channel, target_height, target_width]
        """
        # 1. 统一所有特征图的空间尺寸
        resized_maps = [F.interpolate(f_map, size=(self.target_height, self.target_width), mode='bilinear', align_corners=False)
                        for f_map in feature_maps]
        
        # 2. 在通道维度上拼接
        combined_map = torch.cat(resized_maps, dim=1)  # [batch, total_channels, target_height, target_width]
        # print("Before reduction:", combined_map.shape)

        # 3. 使用 1x1 卷积降维到 mid_channel
        reduced_map = self.channel_reduction(combined_map)
        # print("After reduction:", reduced_map.shape)

        # 4. 获取合并后的特征图的维度信息
        batch, total_channels, height, width = reduced_map.shape
        hw = height * width  # 扁平化的空间维度

        # 5. 重塑为 [batch, total_channels, hw] 以进行计算
        reduced_map_reshaped = reduced_map.view(batch, total_channels, hw)

        # 6. 计算通道关系矩阵 XX^T  -> 形状 [batch, total_channels, total_channels]
        channel_relation = torch.bmm(reduced_map_reshaped, reduced_map_reshaped.transpose(1, 2))

        # 7. 计算三线性注意力图 -> 形状 [batch, total_channels, hw]
        attention_map = torch.bmm(channel_relation, reduced_map_reshaped)

        # 8. 重新恢复到原始空间尺寸 -> 形状 [batch, total_channels, height, width]
        output = attention_map.view(batch, total_channels, height, width)

        return output

class FeatureCompression(nn.Module):
    def __init__(self, in_channels, mid_channels=512, target_size=13):
        super(FeatureCompression, self).__init__()
        
        # Step 1: 使用 3x3 卷积将空间大小从 52x52 降到 13x13
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, 
                              kernel_size=3, stride=4, padding=1)  # Stride=4 以缩小尺寸
        
        # Step 2: 使用 Adaptive Pooling 将尺寸进一步缩小到 1x1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # print("Input shape:", x.shape)  # Step 0
        x = self.conv(x)
        # print("After Conv shape:", x.shape)  # Step 1
        x = self.adaptive_pool(x)
        # print("After Adaptive Pool shape:", x.shape)  # Step 2
        return x



class TrilliAttModules(nn.Module):
    def __init__(self, target_height, target_width, mid_channel, input_layers=4, compression_out_channels=512, compression_target_size=1):
        """
        组合 TrilinearAttention 和 FeatureCompression 的模块
        :param target_height: 目标特征图高度 (用于 TrilinearAttention)
        :param target_width: 目标特征图宽度 (用于 TrilinearAttention)
        :param mid_channel: TrilinearAttention 输出的通道数
        :param input_layers: 输入的不同特征图数量
        :param compression_out_channels: FeatureCompression 目标降维的通道数
        :param compression_target_size: FeatureCompression 最终输出的空间大小 (默认为 1，即 1x1)
        """
        super(TrilliAttModules, self).__init__()

        # Trilinear Attention 模块
        self.trilinear_attention = TrilinearAttention(target_height, target_width, mid_channel, input_layers)

        # Feature Compression 模块
        self.feature_compression = FeatureCompression(in_channels=mid_channel, mid_channels=compression_out_channels, target_size=compression_target_size)

    def forward(self, feature_maps):
        """
        前向传播过程：
        1. 通过 TrilinearAttention 计算融合注意力特征
        2. 通过 FeatureCompression 进行降维，得到最终紧凑的表征
        :param feature_maps: List[Tensor] -> 输入的多个特征图, 每个形状为 [batch, channels, h, w]
        :return: [batch, compression_out_channels, 1, 1] 的最终压缩特征
        """
        # Step 1: 计算三线性注意力融合的特征图
        attention_map = self.trilinear_attention(feature_maps)

        # Step 2: 进行特征压缩，得到最终的 1x1 特征
        compressed_feature = self.feature_compression(attention_map)

        return compressed_feature
    
# 测试代码
# if __name__ == "__main__":
#     # 生成不同尺寸的输入特征图
#     f1 = torch.randn(2, 64, 52, 52)   # 低级特征
#     f2 = torch.randn(2, 128, 28, 28)  # 中级特征
#     f3 = torch.randn(2, 256, 14, 14)  # 高级特征
#     f4 = torch.randn(2, 512, 7, 7)    # 更高级特征

#     # 组合到列表
#     feature_maps = [f1, f2, f3, f4]

#     # 创建 TrAttModules 实例
#     tr_att_module = TrilliAttModules(target_height=52, target_width=52, mid_channel=256, input_layers=4, compression_out_channels=512, compression_target_size=1)

#     # 执行前向传播
#     output = tr_att_module(feature_maps)

#     # 输出最终形状
#     print("Final Output shape:", output.shape)  # 预期: [batch, 512, 1, 1]

# # 示例：输入多个特征图
# f1 = torch.randn(2, 64, 52, 52)  # 特征图 f_1
# f2 = torch.randn(2, 128, 28, 28)  # 特征图 f_2
# f3 = torch.randn(2, 256, 14, 14)  # 特征图 f_3
# f4 = torch.randn(2, 512, 7, 7)
# # 目标降维通道数
# mid_channel = 512

# # 创建 TrilinearAttention 实例，目标尺寸为 52x52，通道降维到 mid_channel
# trilinear_attention = TrilinearAttention(target_height=52, target_width=52, mid_channel=mid_channel)
# fc = FeatureCompression(in_channels=512,mid_channels=512)
# adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

# # 计算注意力增强后的特征图
# output_tensor = trilinear_attention([f1, f2, f3,f4])

# # 输出结果形状
# print("Final Output shape:", fc(output_tensor).shape)