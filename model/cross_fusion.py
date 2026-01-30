import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CrossModalFusion(nn.Module):
    def __init__(self, visual_dim, physical_dim):
        super().__init__()
        
        self.visual_proj = nn.Conv2d(visual_dim, visual_dim, 1)  
        self.physical_proj = nn.Conv2d(physical_dim, visual_dim, 1)  

        self.attention = nn.Sequential(
            nn.Conv2d(visual_dim*2, visual_dim // 8, 1),  
            nn.ReLU(),
            nn.Conv2d(visual_dim // 8, visual_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x_visual, x_physical):
        
        v = self.visual_proj(x_visual)  
        p = self.physical_proj(x_physical)  

        # 跨模态交互
        fusion = torch.cat([v, p], dim=1)  
        attn = self.attention(fusion)  

        # 门控融合
        return x_visual * (1 - attn) + (v * p) * attn  


# 物理特征提取:PFE Block
def extract_physical_features(img_ir):
    """
    img_ir: 原始红外图像张量，值域[0,1]（需先归一化），形状 [B,C,H,W]
    返回：物理特征张量 [B,4,H,W] (固定为4通道)
    """
    # B, C, H, W = img_ir.shape

    ####################################################################################################################################
    # 计算局部均值（mean）：
    mean = F.avg_pool2d(img_ir, 3, stride=1, padding=1).mean(dim=1, keepdim=True)

    # 计算局部标准差（std）：
    std = torch.sqrt(F.avg_pool2d(img_ir**2, 3, stride=1, padding=1).mean(dim=1, keepdim=True) - mean**2)

    # 计算局部信噪比（SNR）：
    snr = (img_ir.mean(dim=1, keepdim=True) - mean) / (std + 1e-6)  # [B,1,H,W]

    # # 最终特征拼接
    return torch.cat([snr], dim=1)  # [B,1,H,W]



class AsymBiChaFuse(nn.Module):
    def __init__(self, channels=64, r=4):
        super(AsymBiChaFuse, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // r)


        self.topdown = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(in_channels=self.channels, out_channels=self.bottleneck_channels, kernel_size=1, stride=1,padding=0),
        nn.BatchNorm2d(self.bottleneck_channels,momentum=0.9),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=self.bottleneck_channels,out_channels=self.channels,  kernel_size=1, stride=1,padding=0),
        nn.BatchNorm2d(self.channels,momentum=0.9),
        nn.Sigmoid()
        )

        self.bottomup = nn.Sequential(
        nn.Conv2d(in_channels=self.channels,out_channels=self.bottleneck_channels, kernel_size=1, stride=1,padding=0),
        nn.BatchNorm2d(self.bottleneck_channels,momentum=0.9),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=self.bottleneck_channels,out_channels=self.channels, kernel_size=1, stride=1,padding=0),
        nn.BatchNorm2d(self.channels,momentum=0.9),
        nn.Sigmoid()
        )

        self.post = nn.Sequential(
        nn.Conv2d(in_channels=channels,out_channels=channels, kernel_size=3, stride=1, padding=1, dilation=1),
        nn.BatchNorm2d(channels,momentum=0.9),
        nn.ReLU(inplace=True)
        )

    def forward(self, xh, xl):

        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl)
        xs = 2 * torch.mul(xl, topdown_wei) + 2 * torch.mul(xh, bottomup_wei)
        xs = self.post(xs)
        return xs