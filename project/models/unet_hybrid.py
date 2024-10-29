import torch.nn as nn
from unet_urpc import Encoder

class MSDNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(MSDNet, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 0,
                  'acti_func': 'relu'}
        
        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)
        self.decoder2 = Decoder(params2)
        
    def forward(self, x):
        shape = x.shape[2:]

        x0, x1, x2, x3, x4 = self.encoder(x)

        d1_out1, d1_out2, d1_ou3, d1_out4 = self.decoder1(x0, x1, x2, x3, x4, shape)
        d2_out1, d2_out2, d2_ou3, d2_out4 = self.decoder2(x0, x1, x2, x3, x4, shape)

        return [d1_out1, d1_out2, d1_ou3, d1_out4], [d2_out1, d2_out2, d2_ou3, d2_out4]
    

class UNet_URPC(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_URPC, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        

        self.encoder = Encoder(params)
        self.decoder = Decoder_URPC(params)

    def forward(self, x):
        shape = x.shape[2:]

        x0, x1, x2, x3, x4 = self.encoder(x)

        dp1_out_seg, dp2_out_seg, dp3_out_seg, dp4_out_seg = self.decoder(x0, x1, x2, x3, x4, shape)
        
        return dp1_out_seg, dp2_out_seg, dp3_out_seg, dp4_out_seg