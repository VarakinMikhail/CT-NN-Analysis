import torch
import torch.nn as nn

#Небольшая нейросеть U-Net для тестирования работоспособности и использования в качестве бейзлайна
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
        x = torch.cat([x, skip_connection], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class MiniUnet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, blocks = 64):
        super().__init__()

        self.encoder1 = EncoderBlock(in_channels, blocks)
        self.encoder2 = EncoderBlock(blocks, 2*blocks)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(2*blocks, 4*blocks, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*blocks),
            nn.ReLU(inplace=True)
        )

        self.decoder1 = DecoderBlock(4*blocks, 2*blocks, 2*blocks)
        self.decoder2 = DecoderBlock(2*blocks, blocks, blocks)

        self.final_conv = nn.Conv2d(blocks, out_channels, kernel_size=1)


    def forward(self, x):
        x, skip1 = self.encoder1(x)
        x, skip2 = self.encoder2(x)

        x = self.bottleneck(x)

        x = self.decoder1(x, skip2)
        x = self.decoder2(x, skip1)

        x = self.final_conv(x)
        return x
