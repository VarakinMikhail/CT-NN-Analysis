import torch
import torch.nn as nn
import torch.nn.functional as F

#Transformer U-Net

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, channels):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(channels)-1):
            self.layers.append(nn.Sequential(
                ConvBlock(channels[i], channels[i+1]),
                ConvBlock(channels[i+1], channels[i+1])
            ))
    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
            x = F.max_pool2d(x, kernel_size=2)
        return x, features

class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(channels)-1):
            self.layers.append(nn.Sequential(
                nn.ConvTranspose2d(channels[i], channels[i+1], kernel_size=2, stride=2),
                ConvBlock(channels[i], channels[i+1]),
                ConvBlock(channels[i+1], channels[i+1])
            ))
    def forward(self, x, features):
        for i in range(len(self.layers)):
            x = self.layers[i][0](x)
            x = torch.cat([x, features[-(i+1)]], dim=1)
            x = self.layers[i][1](x)
            x = self.layers[i][2](x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim*4, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x_res = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x_res + x
        x_res = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x_res + x
        return x

class TransUNet(nn.Module):
    def __init__(self, img_size=128, in_channels=1, num_classes=4, blocks=64, num_heads=8, num_layers=12, embed_dim=512, patch_size=8):
        super(TransUNet, self).__init__()
        self.encoder = Encoder([in_channels, blocks, blocks*2, blocks*4, blocks*8])
        self.bottleneck = nn.Sequential(
            ConvBlock(blocks*8, blocks*16),
            ConvBlock(blocks*16, blocks*16)
        )
        self.patch_embedding = nn.Conv2d(blocks*16, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // (2**4 * patch_size))**2
        
        self.transformer = nn.Sequential(*[TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
        
        self.proj_back = nn.ConvTranspose2d(embed_dim, blocks*16, kernel_size=patch_size, stride=patch_size)
        
        self.decoder = Decoder([blocks*16, blocks*8, blocks*4, blocks*2, blocks])
        
        self.final_conv = nn.Conv2d(blocks, num_classes, kernel_size=1)
    def forward(self, x):
        
        x, features = self.encoder(x)
        
        x = self.bottleneck(x)
        
        B, C, H, W = x.shape
        x = self.patch_embedding(x).flatten(2).permute(2, 0, 1)
        
        x = self.transformer(x)
        
        x = x.permute(1, 2, 0).view(B, -1, H // self.patch_embedding.kernel_size[0], W // self.patch_embedding.kernel_size[0])
        x = self.proj_back(x)
        
        x = self.decoder(x, features)
        
        x = self.final_conv(x)
        return x