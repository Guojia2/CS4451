# TSMixer for multivariate forecasting (arXiv:2303.06053)
# follows batch/sequence conventions
# returns tensors compatible with FreDF training
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class Norm2D(nn.Module):
    # normalizes each sample across time and feature dims

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(-2, -1), keepdim=True)
        var = x.var(dim=(-2, -1), keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps)


class TimeMixingMLP(nn.Module):
    # shared mlp that mixes info along temporal dimension

    def __init__(self, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, c = x.shape
        x = x.reshape(b * c, l)
        x = self.net(x)
        return x.reshape(b, l, c)


class FeatureMixingMLP(nn.Module):
    # shared mlp that mixes info across features

    def __init__(self, enc_in: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(enc_in, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, enc_in),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, c = x.shape
        x = x.reshape(b * l, c)
        x = self.net(x)
        return x.reshape(b, l, c)


class MixerBlock(nn.Module):
    # alternates time and feature mixing w/ residual connections

    def __init__(self, seq_len: int, enc_in: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm_time = Norm2D()
        self.norm_feat = Norm2D()
        self.time_mlp = TimeMixingMLP(seq_len, dropout)
        self.feature_mlp = FeatureMixingMLP(enc_in, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.time_mlp(self.norm_time(x))
        x = x + self.feature_mlp(self.norm_feat(x))
        return x


class TSMixer(nn.Module):
    # tsmixer (arXiv:2303.06053) for multivariate forecasting, I/O [B, L, C]
    # input: B = batch size, number of samples passed through the model at once
    # input: L = lookback or sequence length, number of time steps in the input window
    # input C = channels or variables, number of input series per time step
    # output [B, pred_len, C]
    # pred_len is the number of time steps to predict
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        dec_in: Optional[int] = None,
        d_model: int = 128,
        n_blocks: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.dec_in = dec_in or enc_in
        self.mixer_blocks = nn.ModuleList(
            [MixerBlock(seq_len, enc_in, d_ff, dropout) for _ in range(n_blocks)]
        )
        # temporal projection maps lookback window L to forecast horizon T per feature
        self.temporal_projection = nn.Linear(seq_len, pred_len)
        
        # need this in order to process the f64 values from some datasets
        self.double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if x.dim() != 3:
            raise ValueError(f"TSMixer expects 3D input, got shape {x.shape}")
        # added this for flexibility with the axis ordering, we can process tensors of shape [B,L,C] or [B,C,L], for less trouble with the dataloader
        # after this step, input is guaranteed to be of shape [B,L,C]
        if x.shape[1:] == (self.seq_len, self.enc_in):
            seq_first = True
        elif x.shape[1:] == (self.enc_in, self.seq_len):
            x = x.transpose(1, 2)
            seq_first = False
        else:
            raise ValueError(
                f"Expected shape [B, {self.seq_len}, {self.enc_in}] or [B, {self.enc_in}, {self.seq_len}], got {x.shape}"
            )

        for block in self.mixer_blocks:
            x = block(x)

        b, l, c = x.shape
        temporal = x.transpose(1, 2).reshape(b * c, l)
        temporal = self.temporal_projection(temporal)
        out = temporal.view(b, c, self.pred_len).transpose(1, 2)

        if not seq_first:
            out = out.transpose(1, 2)

        return out


def _shape_test() -> None:
    seq_len, pred_len, enc_in = 32, 8, 5
    model = TSMixer(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in)
    dummy = torch.randn(4, seq_len, enc_in)
    out = model(dummy)
    print("Output shape:", out.shape)


if __name__ == "__main__":
    _shape_test()
