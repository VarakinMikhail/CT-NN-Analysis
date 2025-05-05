import torch
import torch.nn as nn

#Классический U-Net
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        skip_connection = x #перед relu, чтобы сохранить детали
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x, skip_connection

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        x = torch.cat([x, skip_connection], dim=1) #стакнули изображения
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, blocks=64):
        super().__init__()

        self.encoder1 = EncoderBlock(in_channels, blocks)
        self.encoder2 = EncoderBlock(blocks, 2*blocks)
        self.encoder3 = EncoderBlock(2*blocks, 4*blocks)
        self.encoder4 = EncoderBlock(4*blocks, 8*blocks)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(8*blocks, 16*blocks, kernel_size=3, padding=1),
            nn.BatchNorm2d(16*blocks),
            nn.ReLU(inplace=True),
            nn.Conv2d(16*blocks, 16*blocks, kernel_size=3, padding=1),
            nn.BatchNorm2d(16*blocks),
            nn.ReLU(inplace=True)
        )

        self.decoder1 = DecoderBlock(16*blocks, 8*blocks, 8*blocks)
        self.decoder2 = DecoderBlock(8*blocks, 4*blocks, 4*blocks)
        self.decoder3 = DecoderBlock(4*blocks, 2*blocks, 2*blocks)
        self.decoder4 = DecoderBlock(2*blocks, blocks, blocks)

        self.final_conv = nn.Conv2d(blocks, out_channels, kernel_size=1)

    def forward(self, x):
        x, skip1 = self.encoder1(x)
        x, skip2 = self.encoder2(x)
        x, skip3 = self.encoder3(x)
        x, skip4 = self.encoder4(x)

        x = self.bottleneck(x)

        x = self.decoder1(x, skip4)
        x = self.decoder2(x, skip3)
        x = self.decoder3(x, skip2)
        x = self.decoder4(x, skip1)

        x = self.final_conv(x)
        return x