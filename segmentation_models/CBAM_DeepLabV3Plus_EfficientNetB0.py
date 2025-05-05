import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn.functional as F

#Использование EfficientNetB0 вместо MobileNetV2

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)
        channel_attn = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_attn
        
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = torch.cat([avg_out, max_out], dim=1)
        spatial_attn = self.conv_spatial(spatial_attn)
        spatial_attn = self.sigmoid_spatial(spatial_attn)
        x = x * spatial_attn
        
        return x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[2, 4, 6]):
        super(ASPP, self).__init__()
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[0], dilation=rates[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[1], dilation=rates[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[2], dilation=rates[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        
        self.cbam = CBAM(out_channels)
        
    def forward(self, x):
        size = x.shape[2:]
        aspp1 = self.aspp1(x)
        aspp2 = self.aspp2(x)
        aspp3 = self.aspp3(x)
        aspp4 = self.aspp4(x)
        global_avg = self.global_avg_pool(x)
        global_avg = F.interpolate(global_avg, size=size, mode='bilinear', align_corners=True)
        concat = torch.cat([aspp1, aspp2, aspp3, aspp4, global_avg], dim=1)
        concat = self.conv1(concat)
        concat = self.bn1(concat)
        concat = self.relu(concat)
        concat = self.dropout(concat)
        
        concat = self.cbam(concat)
        
        return concat

class Decoder(nn.Module):
    def __init__(self, low_level_in, low_level_out, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_in, low_level_out, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(low_level_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(low_level_out + out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(0.5)
        
        self.cbam = CBAM(out_channels)
        
    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.conv2(x)
        x = self.dropout(x)
        
        x = self.cbam(x)
        
        return x

class CBAM_DeepLabV3Plus_EfficientNetB0(nn.Module):
    def __init__(self, num_classes=4, pretrained=False):
        super(CBAM_DeepLabV3Plus_EfficientNetB0, self).__init__()
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        
        self.adapter = nn.Sequential( 
            nn.Conv2d(1, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        #Извлечение необходимых слоёв из EfficientNet_B0
        self.backbone = create_feature_extractor(
            efficientnet, 
            return_nodes={
                'features.8': 'high_level',  #Последний слой перед классификатором
                'features.2': 'low_level'    #Ранний слой с 24 каналами для декодера
            }
        )
        self.aspp = ASPP(in_channels=1280, out_channels=256, rates=[1, 2, 3])  #Измененные rates для более узкого контекста (или меньших изображений)
        self.decoder = Decoder(low_level_in=24, low_level_out=48, out_channels=256)
        self.final_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        x = self.adapter(x)
        features = self.backbone(x)
        high_level = features['high_level']
        low_level = features['low_level']
        
        aspp_out = self.aspp(high_level)
        decoder_out = self.decoder(aspp_out, low_level)
        out = self.final_conv(decoder_out)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        return out