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

# NOTE: this assumes the compute weighted combined MSE and MAE Loss in the frequency domain.
def fourier_mse_loss(pred: torch.Tensor, target: torch.Tensor, fourier_weight: float) -> torch.Tensor:
    """

    Compute weighted combined MSE and MAE Loss in the frequency domain.
        args:
        pred (Tensor): Predicted COMPLEX values of shape (batches, covariates, freq_bins).
        target (Tensor): True COMPLEX values of shape (batches, covariates, freq_bins).
        fourier_weight (float): Weighting factor for the MAE (l1) component (0 - 1.0).

    Returns:
        tensor: Weighted combined MSE and MAE loss on magnitudes
    """
    assert 0.0 <= fourier_weight <= 1.0, "fourier_weight must be between 0 and 1.0"
    assert pred.shape == target.shape, "Prediction and target shapes must match"
    assert pred.dim() == 3, "Pred and target must be 3D tensors of shape (batches, covariates, freq_bins)"
    # Ensure inputs are complex for these calculations
    assert pred.is_complex() and target.is_complex(), "Inputs to fourier_mse_loss must be complex tensors"

    # Get the magnitudes (absolute values) of the complex frequency components
    pred_magnitude = torch.abs(pred)
    target_magnitude = torch.abs(target)

    # Compute MSE loss on the magnitudes
    mse_mag_loss = F.mse_loss(pred_magnitude, target_magnitude)

    # Compute MAE (L1) loss on the magnitudes (this aligns with the original fourier_loss's logic)
    mae_mag_loss = F.l1_loss(pred_magnitude, target_magnitude)

    # Combine MSE and MAE on magnitudes
    total_loss = (1 - fourier_weight) * mse_mag_loss + fourier_weight * mae_mag_loss
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

 
