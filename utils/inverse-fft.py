
# layer definition for the FFT.
import torch 
import torch.fft as fft 
from torch import nn

class IFFT(nn.Module):
    def __init__(self):
        super(IFFT, self).__init__()


    def forward(self, x):
        return fft.ifft(x, dim=-1).real() # return real part only
        # inverse transform produces complex output
        # take .real to recover real-valued signal


