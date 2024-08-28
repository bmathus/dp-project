from torch import nn
import torch

class UNet(nn.Module):
    def __init__(self,img_channels,num_classes):
        super().__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(img_channels,64,(3,3),stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,(3,3),stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.encoder_block1 = EncoderBlock(64,128)
        self.encoder_block2 = EncoderBlock(128,256)
        self.encoder_block3 = EncoderBlock(256,512)
        self.encoder_block4 = EncoderBlock(512,1024)

        self.decoder_block4 = DecoderBlock(1024,512)
        self.decoder_block3 = DecoderBlock(512,256)
        self.decoder_block2 = DecoderBlock(256,128)
        self.decoder_block1 = DecoderBlock(128,64)

        self.conv_1x1 = nn.Conv2d(64,num_classes,kernel_size=(1,1))

    def forward(self, x):
        # Our model now returns logits!
        out1 = self.input_conv(x)
        out2 = self.encoder_block1(out1)
        out3 = self.encoder_block2(out2)
        out4 = self.encoder_block3(out3)
        out5 = self.encoder_block4(out4)

        x = self.decoder_block4(out5,out4)
        x = self.decoder_block3(x,out3)
        x = self.decoder_block2(x,out2)
        x = self.decoder_block1(x,out1)

        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(in_channels,out_channels,(3,3),stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,(3,3),stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=(2,2),stride=2)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,(3,3),stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,(3,3),stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up_conv(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv_block(x)

        
