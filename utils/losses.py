import torch
import torch.nn.functional as F


def fredf_loss(pred: torch.Tensor, target: torch.Tensor, fourier_weight: float) -> torch.Tensor:
    # fredf loss from paper: lambda * L_freq + (1 - lambda) * L_temp
    # 
    # implementation per the fredf paper (wang et al 2025)
    # -temporal loss: mse in time domain
    # -frequency loss: l1 on fft coefficients  
    # -combined with weighting parameter lambda (fourier_weight)
    # 
    # arg:
    #  pred: time-domain predictions (batches, covariates, forecast_horizon)
    #  target: time-domain targets (batches, covariates, forecast_horizon)
    #  fourier_weight: lambd in [0, 1], weight for frequency loss
    # 
    # returns:
    #     combined loss: lambda * L_freq + (1 - lambda) * L_temp

    assert 0.0 <= fourier_weight <= 1.0, "fourier_weight must be between 0 and 1.0"
    assert pred.shape == target.shape, "Prediction and target shapes must match"
    assert pred.dim() == 3, "Pred and target must be 3D tensors of shape (batches, covariates, forecast_horizon)"

    # temporal loss (mse) - fredf paper eq. 1
    temporal_loss = F.mse_loss(pred, target)
    
    # frequency loss (l1 on fft) - fredf paper eq. 3

    pred_fft = torch.fft.rfft(pred, dim=-1)
    target_fft = torch.fft.rfft(target, dim=-1)

    freq_loss = torch.mean(torch.abs(pred_fft - target_fft))
    
    # combined loss: λ * L_freq + (1 - λ) * L_temp - fredf paper eq. 4
    total_loss = fourier_weight * freq_loss + (1 - fourier_weight) * temporal_loss
    
    return total_loss

 
