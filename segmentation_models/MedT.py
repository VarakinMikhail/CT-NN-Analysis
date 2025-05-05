import torch
import torch.nn as nn

#Использование архитектуры MedT - Medical Transformer

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class TransformerBlock(nn.Module):
    def __init__(self, in_channels, num_heads=8, num_layers=4):
        super(TransformerBlock, self).__init__()
        self.layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=in_channels, nhead=num_heads) for _ in range(num_layers)]
        )
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(2, 0, 1)
        for layer in self.layers:
            x = layer(x)
        x = x.permute(1, 2, 0).view(B, C, H, W)
        return x

class MedT(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, blocks=32):
        super(MedT, self).__init__()
        self.encoder1 = ConvBlock(in_channels, 2*blocks)
        self.pool1 = nn.MaxPool2d(2)
        
        self.encoder2 = ConvBlock(2*blocks, 4*blocks)
        self.pool2 = nn.MaxPool2d(2)
        
        self.encoder3 = ConvBlock(4*blocks, 8*blocks)
        self.pool3 = nn.MaxPool2d(2)
        
        self.encoder4 = ConvBlock(8*blocks, 16*blocks)
        self.pool4 = nn.MaxPool2d(2)
        
        #Transformer block
        self.transformer = TransformerBlock(16*blocks, num_heads=8, num_layers=4)
        
        self.upconv4 = nn.ConvTranspose2d(16*blocks, 8*blocks, kernel_size=2, stride=2)
        self.decoder4 = ConvBlock(24*blocks, 8*blocks) 
        
        self.upconv3 = nn.ConvTranspose2d(8*blocks, 4*blocks, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(12*blocks, 4*blocks)
        
        self.upconv2 = nn.ConvTranspose2d(4*blocks, 2*blocks, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(6*blocks, 2*blocks) 
        
        self.upconv1 = nn.ConvTranspose2d(2*blocks, blocks, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(3*blocks, blocks) 
        
        self.conv_last = nn.Conv2d(blocks, num_classes, kernel_size=1)
        
    def forward(self, x):
        x1 = self.encoder1(x)       
        x_pool1 = self.pool1(x1)    
        
        x2 = self.encoder2(x_pool1) 
        x_pool2 = self.pool2(x2)    
        
        x3 = self.encoder3(x_pool2)   
        x_pool3 = self.pool3(x3)   
        
        x4 = self.encoder4(x_pool3)
        x_pool4 = self.pool4(x4)   
        
        #Transformer
        x_transformed = self.transformer(x_pool4)
        
        x_up4 = self.upconv4(x_transformed)   
        x_cat4 = torch.cat([x_up4, x4], dim=1)
        x_dec4 = self.decoder4(x_cat4)        
        
        x_up3 = self.upconv3(x_dec4)          
        x_cat3 = torch.cat([x_up3, x3], dim=1)
        x_dec3 = self.decoder3(x_cat3)        
        
        x_up2 = self.upconv2(x_dec3)          
        x_cat2 = torch.cat([x_up2, x2], dim=1)  
        x_dec2 = self.decoder2(x_cat2)            
        
        x_up1 = self.upconv1(x_dec2)           
        x_cat1 = torch.cat([x_up1, x1], dim=1)     
        x_dec1 = self.decoder1(x_cat1)          
        
        output = self.conv_last(x_dec1)           
        return output