import torch
import torch.nn as nn

#Четвертая версия Attention U-Net++ с механизмом внимания (увеличение глубины нейронной сети)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        out += residual
        out = self.relu(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class NestedDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, *inputs):
        inputs = [x for x in inputs if x is not None]
        x = torch.cat(inputs, dim=1)
        residual = self.shortcut(x)
        out = self.conv(x)
        out += residual
        out = self.relu(out)
        return out

class AttentionUnetPlusPlus4(nn.Module):
    def __init__(self, in_channels=1, out_channels=8, blocks=64):
        super().__init__()

        nb_filter = [blocks, 2*blocks, 4*blocks, 8*blocks, 16*blocks]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = EncoderBlock(in_channels, nb_filter[0])
        self.conv1_0 = EncoderBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = EncoderBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = EncoderBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = EncoderBlock(nb_filter[3], nb_filter[4])

        # Блоки внимания
        self.attention1 = AttentionBlock(F_g=nb_filter[1], F_l=nb_filter[0], F_int=nb_filter[0]//2)
        self.attention2 = AttentionBlock(F_g=nb_filter[2], F_l=nb_filter[1], F_int=nb_filter[1]//2)
        self.attention3 = AttentionBlock(F_g=nb_filter[3], F_l=nb_filter[2], F_int=nb_filter[2]//2)
        self.attention4 = AttentionBlock(F_g=nb_filter[4], F_l=nb_filter[3], F_int=nb_filter[3]//2)

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
        # Этап энкодера
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))

        # Внимание и декодер
        x0_0_att = self.attention1(g=self.up(x1_0), x=x0_0)
        x0_1 = self.conv0_1(torch.cat([x0_0_att, self.up(x1_0)], dim=1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_0_att = self.attention2(g=self.up(x2_0), x=x1_0)
        x1_1 = self.conv1_1(torch.cat([x1_0_att, self.up(x2_0)], dim=1))

        x0_1_att = self.attention1(g=self.up(x1_1), x=x0_1)
        x0_2 = self.conv0_2(torch.cat([x0_0_att, x0_1_att, self.up(x1_1)], dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_0_att = self.attention3(g=self.up(x3_0), x=x2_0)
        x2_1 = self.conv2_1(torch.cat([x2_0_att, self.up(x3_0)], dim=1))

        x1_1_att = self.attention2(g=self.up(x2_1), x=x1_1)
        x1_2 = self.conv1_2(torch.cat([x1_0_att, x1_1_att, self.up(x2_1)], dim=1))

        x0_2_att = self.attention1(g=self.up(x1_2), x=x0_2)
        x0_3 = self.conv0_3(torch.cat([x0_0_att, x0_1_att, x0_2_att, self.up(x1_2)], dim=1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_0_att = self.attention4(g=self.up(x4_0), x=x3_0)
        x3_1 = self.conv3_1(torch.cat([x3_0_att, self.up(x4_0)], dim=1))

        x2_1_att = self.attention3(g=self.up(x3_1), x=x2_1)
        x2_2 = self.conv2_2(torch.cat([x2_0_att, x2_1_att, self.up(x3_1)], dim=1))

        x1_2_att = self.attention2(g=self.up(x2_2), x=x1_2)
        x1_3 = self.conv1_3(torch.cat([x1_0_att, x1_1_att, x1_2_att, self.up(x2_2)], dim=1))

        x0_3_att = self.attention1(g=self.up(x1_3), x=x0_3)
        x0_4 = self.conv0_4(torch.cat([x0_0_att, x0_1_att, x0_2_att, x0_3_att, self.up(x1_3)], dim=1))

        output = self.final(x0_4)
        return output