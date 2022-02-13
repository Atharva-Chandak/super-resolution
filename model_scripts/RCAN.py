from turtle import forward
import torch
import torch.nn as nn
from utils.norm import Norm
from utils.upsampler import Upsample


# Channel Attention Module
class ChannelAttention(nn.Module):

    def __init__(self,  n_feats = 256, reduction_factor = 16 ):

        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(n_feats, n_feats // reduction_factor, kernel_size = 1 ),
                nn.ReLU(True),
                nn.Conv2d(n_feats // reduction_factor, n_feats, kernel_size = 1 ),
                nn.Sigmoid()
        )

    def forward(self, x):

        mask = self.attention(x)   
        out = mask * x

        return out


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):

    def __init__( self, n_feats = 64, block_size = 2, kernel_size = 3, reduction_factor = 16, activation = nn.ReLU(True), res_scale_factor = 1 ):

        super(RCAB, self).__init__()
        self.res_scale_factor = res_scale_factor
        
        modules = []
        for i in range(block_size - 1):
            modules.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding = kernel_size//2 ))
            modules.append(activation)
        modules.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding = kernel_size//2 ))
        
        modules.append( ChannelAttention( n_feats = n_feats, reduction_factor = reduction_factor ))
        self.body = nn.Sequential( *modules )

    def forward(self, x):

        out = self.body(x)*self.res_scale_factor
        out += x
        
        return out


# Residual Group (ResGroup)
class ResGroup(nn.Module):

    def __init__(self, n_resblocks = 20, block_size = 2, n_feats = 64, kernel_size = 3, reduction_factor = 16, \
            activation=nn.ReLU(True), res_scale_factor = 1, 
        ):
        super(ResGroup, self).__init__()
        modules = []
        for _ in range(n_resblocks):
            modules.append(
                RCAB(n_feats = n_feats, block_size = block_size, kernel_size = kernel_size, reduction_factor = reduction_factor, \
                    activation = activation, res_scale_factor = res_scale_factor )
            )
        modules.append( nn.Conv2d( n_feats, n_feats, kernel_size, padding = kernel_size//2 ))
        self.body = nn.Sequential(*modules)

    def forward(self, x):

        out = self.body(x)
        out += x
        
        return out


# Residual in Residual Block (RIRBlock)
class RIRBlock(nn.Module):

    def __init__( self, n_resgroups = 10, n_resblocks = 20, block_size = 2, n_feats = 64, kernel_size = 3 , \
            activation = nn.ReLU(True), reduction_factor = 16, res_scale_factor = 1 \
        ):
        super(RIRBlock, self).__init__()
        modules = []

        for _ in range(n_resgroups):
            modules.append(
                ResGroup( n_resblocks = n_resblocks, n_feats = n_feats, block_size = block_size, kernel_size = kernel_size, \
                    reduction_factor = reduction_factor, activation = activation, res_scale_factor = res_scale_factor)
            )
        modules.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2 ))
        self.body = nn.Sequential(*modules)

    def forward( self, x ):

        out = self.body(x)
        out += x
        
        return out

class RCAN(nn.Module):
    """
    Residual Channel Attention Network as described in https://arxiv.org/abs/1807.02758

    Args:
        scale (int): scaling factor of super resolution
        n_resgroups (int): number of Residual Groups in Residual in Residual (RIR) Block 
        n_resblocks (int): number of Residual Channel Attention Blocks in each Residual Group 
        block_size (int): number of conv layers in Residual Block 
        n_feats (int): number of feature maps
        kernel_size (int): size of convolutional kernel
        reduction_factor (int): reduction scale for Channel attention 
        activation (nn.Module): activation function to be used
        res_scale_factor (float): scaling factor inside residual blocks
        rgb_channel_means (torch.tensor): channel wise means (for normalization)
        rgb_channel_stdevs (torch.tensor):channel wise standard deviation (for normalization)

    """
    def __init__(self, scale = 4, n_resgroups = 10, n_resblocks = 20, block_size = 2, n_feats = 64, \
            kernel_size = 3, reduction_factor = 16, activation = nn.ReLU(True), res_scale_factor = 1, \
            rgb_channel_means = torch.tensor([0.4488, 0.4371, 0.4040]), \
            rgb_channel_stdevs = torch.tensor([1., 1., 1.]) \
        ):
        
        super(RCAN, self).__init__()

        self.n_feats = n_feats
        self.scale = scale
        
        # RGB mean for dataset
        self.rgb_channel_means = rgb_channel_means
        self.rgb_channel_stdevs = rgb_channel_stdevs
        self.norm = Norm( self.rgb_channel_means, self.rgb_channel_stdevs)
        self.norm2 = Norm( self.rgb_channel_means, self.rgb_channel_stdevs, sign = 1 )

        self.extractor = nn.Conv2d(3, n_feats, kernel_size, padding=kernel_size//2 )
        
        self.model = RIRBlock( n_resgroups = n_resgroups, n_resblocks = n_resblocks, block_size = block_size, \
                n_feats = n_feats, kernel_size = kernel_size, reduction_factor = reduction_factor, \
                activation = activation, res_scale_factor = res_scale_factor )

        self.upsampler = nn.Sequential(
            Upsample( self.scale, self.n_feats ),
            nn.Conv2d( n_feats, 3, kernel_size, padding = kernel_size//2 )
        )


    def forward( self, x, upsample = True):
        
        x_norm = self.norm(x)
        x_ext = self.extractor(x_norm)

        out = self.model(x_ext)

        if upsample:
            out = self.upsampler(out)
        out = self.norm2(out)

        return out