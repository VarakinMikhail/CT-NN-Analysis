import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

#DeepLabV3+ с экстрактором признаков MobileNetV2 (модифицированной для одноканальных изображений)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
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
    
    def forward(self, x):
        size = x.shape[2:]
        aspp1 = self.aspp1(x)
        aspp2 = self.aspp2(x)
        aspp3 = self.aspp3(x)
        aspp4 = self.aspp4(x)
        global_avg = self.global_avg_pool(x)
        global_avg = nn.functional.interpolate(global_avg, size=size, mode='bilinear', align_corners=True)
        concat = torch.cat([aspp1, aspp2, aspp3, aspp4, global_avg], dim=1)
        concat = self.conv1(concat)
        concat = self.bn1(concat)
        concat = self.relu(concat)
        concat = self.dropout(concat)
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
    
    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        
        x = nn.functional.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.conv2(x)
        x = self.dropout(x)
        return x

class DeepLabV3Plus_MobileNetV2(nn.Module):
    def __init__(self, num_classes=4, pretrained=False):
        super(DeepLabV3Plus_MobileNetV2, self).__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        
        #Модификация первой свёрточной слоя для 1 канала
        first_conv = mobilenet.features[0][0]
        if first_conv.in_channels != 1:
            new_first_conv = nn.Conv2d(
                1,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )
            #Инициализация новых весов путем среднего значения по каналам
            new_first_conv.weight.data = first_conv.weight.data.mean(dim=1, keepdim=True)
            mobilenet.features[0][0] = new_first_conv
        
        #Извлечение необходимых слоёв
        self.backbone = create_feature_extractor(
            mobilenet, 
            return_nodes={
                'features.18': 'high_level',  #Последний слой MobileNetV2
                'features.3': 'low_level'     #Более ранний слой с 24 каналами для декодера
            }
        )
        self.aspp = ASPP(in_channels=1280, out_channels=256)
        self.decoder = Decoder(low_level_in=24, low_level_out=48, out_channels=256)
        self.final_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        high_level = features['high_level']
        low_level = features['low_level']
        
        aspp_out = self.aspp(high_level)
        decoder_out = self.decoder(aspp_out, low_level)
        out = self.final_conv(decoder_out)
        out = nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        return out