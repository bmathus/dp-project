import torch.nn as nn
import torch
from project.models.unet_urpc import FeatureDropout,FeatureNoise,Dropout
from project.models.unet_mcnet import Encoder
from project.models.unet_mcnet import UpBlock


class DecoderMS(nn.Module):
    def __init__(self, params):
        super(DecoderMS, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, mode_upsampling=self.up_type)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,kernel_size=3, padding=1)
        self.feature_noise = FeatureNoise()

    def forward(self, x0, x1, x2, x3, x4, shape):
        x = self.up1(x4, x3)
        if self.training:
            dp3_out_seg = self.out_conv_dp3(Dropout(x, p=0.5))
        else:
            dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        if self.training:
            dp2_out_seg = self.out_conv_dp2(FeatureDropout(x))
        else:
            dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        if self.training:
            dp1_out_seg = self.out_conv_dp1(self.feature_noise(x))
        else:
            dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


class DBPNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(DBPNet, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 2,
                  'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}
        
        self.encoder = Encoder(params1)
        self.decoder1 = DecoderMS(params1)
        self.decoder2 = DecoderMS(params2)
        
    def forward(self, x):
        shape = x.shape[2:]

        x0, x1, x2, x3, x4 = self.encoder(x)

        d1_out1, d1_out2, d1_out3, d1_out4 = self.decoder1(x0, x1, x2, x3, x4, shape)
        d2_out1, d2_out2, d2_out3, d2_out4 = self.decoder2(x0, x1, x2, x3, x4, shape)

        return [d1_out1, d1_out2, d1_out3, d1_out4], [d2_out1, d2_out2, d2_out3, d2_out4]
    

class UNet_DBP(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_DBP, self).__init__()

        params1 = {'in_chns': in_chns,
                'feature_chns': [16, 32, 64, 128, 256],
                'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                'class_num': class_num,
                'up_type': 2,
                'acti_func': 'relu'}
        

        self.encoder = Encoder(params1)
        self.decoder1 = DecoderMS(params1)

    def forward(self, x):
        shape = x.shape[2:]

        x0, x1, x2, x3, x4 = self.encoder(x)

        d1_out1, d1_out2, d1_out3, d1_out4 = self.decoder1(x0, x1, x2, x3, x4, shape)

        return d1_out1, d1_out2, d1_out3, d1_out4
    