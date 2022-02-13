import torch
import torch.nn as nn
from utils.norm import Norm
from utils.upsampler import Upsample

class EDSRResidualBlock(nn.Module):

    def __init__( self, n_feats = 256, block_size = 2, kernel_size = 3 , activation = nn.ReLU(True), res_scale_factor = 1):
    
        super(EDSRResidualBlock, self).__init__()
        self.res_scale_factor = res_scale_factor

        modules=[]
        for i in range(block_size - 1):
            modules.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding = kernel_size//2 ))
            modules.append(activation)
        modules.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding = kernel_size//2 ))
        
        self.body = nn.Sequential(*modules)

    
    def forward(self,x):

        out = self.body(x)*self.res_scale_factor + x 
        
        return out


class EDSR(nn.Module):
    """
    Enhanced Deep Super Resolution network as described in https://arxiv.org/abs/1707.02921

    Args:
        scale (int): scaling factor of super resolution
        n_blocks (int): number of Residual Blocks
        block_size (int): number of conv layers in Residual Block 
        n_feats (int): number of feature maps
        kernel_size (int): size of convolutional kernel
        activation (nn.Module): activation function to be used
        res_scale_factor (int): scaling factor inside residual blocks
        rgb_channel_means (torch.tensor): channel wise means (for normalization)
        rgb_channel_stdevs (torch.tensor):channel wise standard deviation (for normalization)

    """
    def __init__( self, scale = 4, n_blocks = 32, n_feats = 256, kernel_size = 3, \
            activation = nn.ReLU(True), res_scale_factor = 0.1, \
            rgb_channel_means = torch.tensor([0.4488, 0.4371, 0.4040]), \
            rgb_channel_stdevs = torch.tensor([1., 1., 1.]) \
        ):
        
        super(EDSR,self).__init__()

        self.n_feats = n_feats
        self.scale = scale
        
        # RGB mean for dataset
        self.rgb_channel_means = rgb_channel_means
        self.rgb_channel_stdevs = rgb_channel_stdevs
        self.norm = Norm( self.rgb_channel_means, self.rgb_channel_stdevs )
        self.norm2 = Norm( self.rgb_channel_means, self.rgb_channel_stdevs, sign = 1 )
        self.extractor = nn.Conv2d(3, n_feats, kernel_size, padding=kernel_size//2 )

        modules=[]

        for _ in range(n_blocks):
            modules.append(EDSRResidualBlock( n_feats = n_feats, kernel_size = kernel_size, \
                activation = activation, res_scale_factor = res_scale_factor ))
        modules.append( nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2 ) )
        
        self.model = nn.Sequential(*modules)

        self.upsampler = nn.Sequential(
            Upsample( self.scale, self.n_feats ),
            nn.Conv2d( n_feats, 3, kernel_size, padding=kernel_size//2 )
        )

    def forward( self, x, upsample = True):
        
        x_norm = self.norm(x)
        x_ext = self.extractor(x_norm)
        out = self.model(x_ext)+x_ext
        if upsample:
            out = self.upsampler(out)
        out = self.norm2(out)

        return out