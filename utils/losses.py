import torch
import torch.nn.functional as F

def fourier_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # compute Fourier Loss using rfft
    # uses L1 loss on complex FFT values (FreDF paper)
    # pred/target shape:  (batches, covariates, forecast_horizon)
    assert pred.shape == target.shape, "Prediction and target shapes must match"
    assert pred.dim() == 3, "Pred and target must be 3D tensors of shape (batches, covariates, forecast_horizon)"

    # Compute the FFT of predictions and targets along the time dimension
    # Using rfft for real-valued input (more efficient)
    pred_fft = torch.fft.rfft(pred, dim=-1)
    target_fft = torch.fft.rfft(target, dim=-1)

    # Compute L1 loss on complex values
    # The paper uses L1 norm: ||FFT(pred) - FFT(target)||_1
    loss = torch.mean(torch.abs(pred_fft - target_fft))
    
    return loss


def fourier_mse_loss(pred: torch.Tensor, target: torch.Tensor, fourier_weight: float) -> torch.Tensor:
    # weighted Fourier + MSE Loss 
    # FreDF loss: L = λ * L_freq + (1 - λ) * L_temp
    # fourier_weight: λ factor (0 to 1), higher = more frequency emphasis
    assert 0.0 <= fourier_weight <= 1.0, "fourier_weight must be between 0 and 1.0"
    assert pred.shape == target.shape, "Prediction and target shapes must match"
    assert pred.dim() == 3, "Pred and target must be 3D tensors of shape (batches, covariates, forecast_horizon)"

    # Temporal loss (MSE)
    mse_loss = F.mse_loss(pred, target)
    
    # Frequency loss (L1 on FFT)
    f_loss = fourier_loss(pred, target)
    
    # Combined loss: λ * L_freq + (1 - λ) * L_temp
    total_loss = fourier_weight * f_loss + (1 - fourier_weight) * mse_loss
    
    return total_loss


def fourier_mae_loss(pred: torch.Tensor, target: torch.Tensor, fourier_weight: float) -> torch.Tensor:
    # weighted Fourier + MAE Loss
    # alternative using MAE instead of MSE in temporal domain
    assert 0.0 <= fourier_weight <= 1.0, "fourier_weight must be between 0 and 1.0"
    assert pred.shape == target.shape, "Prediction and target shapes must match"
    assert pred.dim() == 3, "Pred and target must be 3D tensors of shape (batches, covariates, forecast_horizon)"

    # Temporal loss (MAE)
    mae_loss = F.l1_loss(pred, target)
    
    # Frequency loss (L1 on FFT)
    f_loss = fourier_loss(pred, target)
    
    # Combined loss: λ * L_freq + (1 - λ) * L_temp
    total_loss = fourier_weight * f_loss + (1 - fourier_weight) * mae_loss
    
    return total_loss

 
