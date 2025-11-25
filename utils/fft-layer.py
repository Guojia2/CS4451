# layer definition for the FFT.
import torch 
import torch.fft as fft 
from torch import nn

class FFTLayer(nn.Module):
    def __init__(self):
        super(FFTLayer, self).__init__()


    def forward(self, x):
        return fft.fft(x, dim=1) 
        # NOTE: fft() is used instead of rfft in case of diverse, complex numbers. rfft() would work more efficiently and only computes positive frequencies.
        # Depending on the nature of the data, the FFT applied should be different. Sticking to fft() for now.
        #
        # - torch.fft.fft(): The general-purpose 1D FFT. Works for both real and complex inputs,
        #   producing a full complex output. Useful when inputs might be complex or
        #   when full spectrum (including negative frequencies) is needed.
        # - torch.fft.rfft(): Real-to-complex 1D FFT. Optimized for real-valued inputs,
        #   returning only the unique positive frequency components (due to Hermitian symmetry).
        #   More efficient for purely real time series.
        # - torch.fft.fftn(): N-dimensional FFT. Applies fft() across multiple specified dimensions.
        #   Used for multi-dimensional data like images (e.g., 2D FFT across height and width).
        # - torch.fft.rfftn(): N-dimensional real-to-complex FFT. Optimized for real-valued
        #   multi-dimensional inputs, returning only unique positive frequency components across
        #   the last transformed dimension. Efficient for real multi-dimensional data.


