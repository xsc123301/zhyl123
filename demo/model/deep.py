import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling模块"""

    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()

        # 1x1卷积
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True)
        )

        # 3x3卷积，扩张率=6
        self.conv3x3_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6),
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True)
        )

        # 3x3卷积，扩张率=12
        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12),
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True)
        )

        # 3x3卷积，扩张率=18
        self.conv3x3_3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18),
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True)
        )

        # 全局平均池化 - 修复BatchNorm问题
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)  # 移除BatchNorm，因为1x1特征图会导致问题
        )

        self.conv1x1_output = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3_1(x)
        x3 = self.conv3x3_2(x)
        x4 = self.conv3x3_3(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1x1_output(x)

        return x


class DeepLabv3Plus(nn.Module):
    """DeepLabv3+模型实现(适配ResNet backbone)"""

    def __init__(self, num_classes=1, backbone='resnet50', pretrained=True):
        super(DeepLabv3Plus, self).__init__()

        # Backbone网络
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            # 修改后的通道数设置
            self.low_level_channels = 256  # layer1输出通道数
            self.high_level_channels = 2048
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            self.low_level_channels = 256
            self.high_level_channels = 2048
        else:
            raise ValueError("Unsupported backbone: {}".format(backbone))

        # 修改ResNet结构 - 移除最后两个block以保持更高分辨率
        self.backbone.layer4 = nn.Sequential(
            *list(self.backbone.layer4.children())[:-2]
        )

        # ASPP模块
        self.aspp = ASPP(in_channels=self.high_level_channels)

        # Decoder部分 - 修改输入通道数
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(512, 48, kernel_size=1),  # 修改为512，因为layer2输出512通道
            nn.BatchNorm2d(48, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True)
        )

        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1),  # 48+256=304
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Encoder部分
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # 获取低级特征
        x_low = self.backbone.layer1(x)
        x_low = self.backbone.layer2(x_low)

        # 获取高级特征
        x_high = self.backbone.layer3(x_low)
        x_high = self.backbone.layer4(x_high)

        # ASPP处理
        x_high = self.aspp(x_high)
        x_high = F.interpolate(x_high, scale_factor=4, mode='bilinear', align_corners=True)

        # 低级特征处理
        x_low = self.decoder_conv1(x_low)

        # 特征融合
        x = torch.cat([x_high, x_low], dim=1)
        x = self.decoder_conv2(x)

        # 输出尺寸调整为1024x1024
        x = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=True)

        return x