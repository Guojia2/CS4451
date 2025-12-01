# quick synthetic training to validate TSMixer + FreDF loss
# generates sine mixtures so no external datasets needed
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.TSMixer import TSMixer
from utils.losses import fourier_mse_loss


@dataclass
class SynthConfig:
    seq_len: int = 96
    pred_len: int = 24
    enc_in: int = 4
    train_samples: int = 256
    batch_size: int = 32
    epochs: int = 20
    fourier_weight: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SineMixDataset(Dataset):
    # produces sine mixture sequences for FreDF

    def __init__(self, cfg: SynthConfig, seed: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self.rng = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.cfg.train_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq_total = self.cfg.seq_len + self.cfg.pred_len
        t = torch.linspace(0, 2 * math.pi, seq_total)
        waves = []
        for _ in range(self.cfg.enc_in):
            freq = torch.rand(1, generator=self.rng) * 0.8 + 0.2
            phase = torch.rand(1, generator=self.rng) * 2 * math.pi
            waves.append(torch.sin(freq * t + phase))
        series = torch.stack(waves, dim=-1)
        noise = torch.randn_like(series) * 0.05
        series = series + noise
        x = series[: self.cfg.seq_len]
        y = series[-self.cfg.pred_len :]
        return x, y


def run_demo(cfg: SynthConfig | None = None) -> None:
    cfg = cfg or SynthConfig()
    device = torch.device(cfg.device)
    dataset = SineMixDataset(cfg)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = TSMixer(
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        enc_in=cfg.enc_in,
        n_blocks=4,
        d_ff=256,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    for epoch in range(1, cfg.epochs + 1):
        total_loss = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = fourier_mse_loss(out, y, fourier_weight=cfg.fourier_weight)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch:02d} | Loss {avg_loss:.4f}")


if __name__ == "__main__":
    run_demo()
