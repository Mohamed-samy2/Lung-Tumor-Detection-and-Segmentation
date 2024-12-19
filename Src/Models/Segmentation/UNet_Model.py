import torch
import torch.nn as nn

class TwoConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(TwoConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding='same')
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(0.2)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding='same')
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(0.2)

    def forward(self,x):
        x=self.conv1(x)
        x=self.batch_norm1(x)
        x=self.relu1(x)
        x = self.dropout1(x)
        x=self.conv2(x)
        x=self.batch_norm2(x)
        x=self.relu2(x)
        x = self.dropout2(x)
        return x
    
class DownBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DownBlock, self).__init__()
        self.two_conv = TwoConv(in_channels,out_channels)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self,x):
        skip_connection = self.two_conv(x)
        down_out = self.down_sample(skip_connection)
        return (down_out,skip_connection)
    

class UpBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(UpBlock, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2)
        self.two_conv = TwoConv(in_channels,out_channels)

    def forward(self,x,skip_connection):
        x = self.up_conv(x)
        x = torch.cat([x, skip_connection], dim=1)
        return self.two_conv(x)
    

class UNet(nn.Module):
    def __init__(self,out_classes=1):
        super(UNet, self).__init__()

        # Encoder
        self.down1=DownBlock(1,32) # out 128
        self.down2=DownBlock(32,64) # out 64
        self.down3=DownBlock(64,128) # out 32
        self.down4=DownBlock(128,256) # out 16
        self.down5=DownBlock(256,512) # out 8
        
        # BottleNeck
        self.two_conv = TwoConv(512,1024) # out 4
        # Decoder
        self.up5 = UpBlock(1024,512) # out 16
        self.up4 = UpBlock(512,256) # out 32
        self.up3 = UpBlock(256,128) # out 64
        self.up2 = UpBlock(128,64) # out 128
        self.up1 = UpBlock(64,32) # out 256
        
        # Output
        self.last_conv = nn.Conv2d(32,out_classes,kernel_size=1)
        
    def forward(self,img):
        x,skip1 = self.down1(img)
        x,skip2 = self.down2(x)
        x,skip3 = self.down3(x)
        x,skip4 = self.down4(x)
        x,skip5 = self.down5(x)
        
        x = self.two_conv(x)
        
        x = self.up5(x,skip5)
        x = self.up4(x,skip4)
        x = self.up3(x,skip3)
        x = self.up2(x,skip2)
        x = self.up1(x,skip1)

        return self.last_conv(x)