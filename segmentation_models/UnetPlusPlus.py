import torch
import torch.nn as nn

#Архитектура U-Net++
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class NestedDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, *inputs):
        inputs = [x for x in inputs if x is not None]
        x = torch.cat(inputs, dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class UnetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, blocks=64):
        super().__init__()

        nb_filter = [blocks, 2*blocks, 4*blocks, 8*blocks, 16*blocks]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = EncoderBlock(in_channels, nb_filter[0])
        self.conv1_0 = EncoderBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = EncoderBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = EncoderBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = EncoderBlock(nb_filter[3], nb_filter[4])

        self.conv0_1 = NestedDecoderBlock(nb_filter[0]+nb_filter[1], nb_filter[0])
        self.conv1_1 = NestedDecoderBlock(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv2_1 = NestedDecoderBlock(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv3_1 = NestedDecoderBlock(nb_filter[3]+nb_filter[4], nb_filter[3])

        self.conv0_2 = NestedDecoderBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        self.conv1_2 = NestedDecoderBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1])
        self.conv2_2 = NestedDecoderBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2])

        self.conv0_3 = NestedDecoderBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0])
        self.conv1_3 = NestedDecoderBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1])

        self.conv0_4 = NestedDecoderBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output