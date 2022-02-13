import torch
import torch.nn as nn

# Normalization Module
class Norm(nn.Conv2d):
    
    def __init__(self, rgb_channel_means, rgb_channel_stdevs, sign=-1):
        
        super(Norm, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).reshape(3, 3, 1, 1)/rgb_channel_stdevs.reshape(3, 1, 1, 1)
        self.bias.data = (sign * rgb_channel_means)/rgb_channel_stdevs
        self.requires_grad = False