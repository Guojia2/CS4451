# unit tests for TSMixer - shape checks, sinusoid fit, FreDF loss compat
import unittest

import torch
import torch.nn as nn

from models.TSMixer import TSMixer
from utils.losses import fredf_loss


def _make_sinusoid(batch: int, total_len: int, enc_in: int) -> torch.Tensor:
    t = torch.linspace(0, 2 * torch.pi, total_len)
    waves = []

    for _ in range(enc_in):
        freq = torch.rand(1) * 1.5 + 0.5
        phase = torch.rand(1) * 2 * torch.pi
        waves.append(torch.sin(freq * t + phase))

    base = torch.stack(waves, dim=-1)
    return base.expand(batch, -1, -1)


class TSMixerUnitTest(unittest.TestCase):
    def test_output_shapes_hold(self) -> None:
        torch.manual_seed(0)

        configs = [
            (32, 12, 4, 2),
            (48, 24, 1, 3),
            (64, 16, 8, 1),
        ]

        for seq_len, pred_len, enc_in, batch in configs:
            with self.subTest(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in, batch=batch):
                model = TSMixer(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in, n_blocks=2)
                x = torch.randn(batch, seq_len, enc_in)
                out = model(x)

                self.assertEqual(out.shape, (batch, pred_len, enc_in))

    def test_can_overfit_small_sinusoid(self) -> None:
        torch.manual_seed(1)
        seq_len, pred_len, enc_in = 50, 10, 1
        model = TSMixer(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in, n_blocks=3, d_ff=128)
        data = _make_sinusoid(batch=64, total_len=seq_len + pred_len, enc_in=enc_in)
        x, y = data[:, :seq_len, :], data[:, -pred_len:, :]

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = nn.MSELoss()

        for epoch in range(200):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        self.assertLess(loss.item(), 0.05)

    def test_fredf_loss_backward(self) -> None:
        torch.manual_seed(2)
        seq_len, pred_len, enc_in = 36, 18, 3

        model = TSMixer(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in)
        x = torch.randn(4, seq_len, enc_in)
        target = torch.randn(4, pred_len, enc_in)
        out = model(x)

        loss = fredf_loss(out, target, fourier_weight=0.4)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]

        self.assertTrue(any(g is not None and torch.isfinite(g).all() for g in grads))


if __name__ == "__main__":
    unittest.main()
