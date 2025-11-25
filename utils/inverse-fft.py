
# layer definition for the FFT.
import torch 
import torch.fft as fft 
from torch import nn

class IFFT(nn.Module):
    def __init__(self):
        super(IFFT, self).__init__()


    def forward(self, x):
        return fft.ifft(x, dim=-1).real() # return only the real part

        # NOTE: The inverse transform will generally produce a complex-valued output.
        # If the original signal was real, you would typically take the real part (e.g., .real)
        # after the inverse transform to recover the real-valued signal.
        #
        # Explanations for the inverse FFT variants:
        #
        # - torch.fft.ifft(): The general-purpose 1D inverse FFT. It performs the inverse
        #   of fft() and typically produces a complex-valued output.
        # - torch.fft.irfft(): The inverse for rfft(). It takes a half-spectrum (as produced
        #   by rfft()) and reconstructs the full real-valued signal. This is very
        #   efficient for reconstructing real-valued time series.
        # - torch.fft.ifftn(): N-dimensional inverse FFT. It performs the inverse of fftn()
        #   across multiple specified dimensions.
        # - torch.fft.irfftn(): N-dimensional inverse for rfftn(). It reconstructs a real-valued
        #   N-D signal from a multi-dimensional half-spectrum.


