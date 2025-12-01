# layer definition for the fft
import torch 
import torch.fft as fft 
from torch import nn

class FFTLayer(nn.Module):
    def __init__(self):
        super(FFTLayer, self).__init__()


    def forward(self, x):
        return fft.fft(x, dim=1)
        # using fft() instead of rfft in case of complex numbers
        # rfft() would be more efficient for real-valued data


