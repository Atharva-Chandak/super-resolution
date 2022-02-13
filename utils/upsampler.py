import torch
import torch.nn as nn
import math

# Upsampling Module
class Upsample(nn.Module):
    def __init__(self, scale, n_feats, kernel_size=3, activation = nn.ReLU(True), b_norm=False):

        super(Upsample, self).__init__()
        modules = []

        if math.log(scale, 2).is_integer():
            for sc in range(int(math.log(scale, 2))):
                modules.append( nn.Conv2d( n_feats, (2 * 2) * n_feats, kernel_size, padding=kernel_size//2 ) )
                modules.append( nn.PixelShuffle(2) )
                if b_norm: 
                    modules.append( nn.BatchNorm2d( n_feats ) )
                # modules.append( activation )

        elif scale == 3:
            modules.append( nn.Conv2d(n_feats, (3*3) * n_feats, kernel_size, padding=kernel_size//2 ) )
            modules.append( nn.PixelShuffle(3) )
            if b_norm: 
                modules.append( nn.BatchNorm2d( n_feats ) )
            # modules.append( activation )

        else:
            print("Scale not implemented")

        self.upsample = nn.Sequential(*modules)

    def forward( self, x):
        
        out = self.upsample(x)

        return out