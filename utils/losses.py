import torch
import torch.nn.functional as F

def fourier_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the Fourier Loss between predictions and targets
    This loss emphasizes differences in the frequency domain.

    Args:
        pred (Tensor): Predicted values of shape (batches, covariates, forecast_horizon).
        target (Tensor): True values of shape (batches, covariates, forecast_horizon)

    Returns:
        Tensor: Discrete Fourier Loss.
    """
    assert pred.shape == target.shape, "Prediction and target shapes must match"
    assert pred.dim() == 3, "Pred and target must be 3D tensors of shape (batches, covariates, forecast_horizon)"

    # Compute the FFT of predictions and targets (rfft for real-valued input so only positive frequencies are returned)
    pred_fft = torch.fft.rfft(pred, dim=-1)
    target_fft = torch.fft.rfft(target, dim=-1)

    # Compute the magnitude spectra
    pred_magnitude = torch.abs(pred_fft)
    target_magnitude = torch.abs(target_fft)

    # compute L1 loss in the frequency domain
    loss = torch.mean(torch.abs(pred_magnitude - target_magnitude))
    return loss

# NOTE: this assumes the compute weighted combined MSE and MAE Loss in the frequency domain.
def fourier_mse_loss(pred: torch.Tensor, target: torch.Tensor, fourier_weight: float) -> torch.Tensor:
    """
    Compute weighted combined MSE and MAE Loss in the frequency domain.
       Args:
        pred (Tensor): Predicted COMPLEX values of shape (batches, covariates, freq_bins).
        target (Tensor): True COMPLEX values of shape (batches, covariates, freq_bins).
        fourier_weight (float): Weighting factor for the MAE (L1) component (0 to 1.0).

    Returns:
        Tensor: Weighted combined MSE and MAE loss on magnitudes.
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


# weighted fourier + MAE loss between inputs and labels. Same args as above.
def fourier_mae_loss(pred: torch.Tensor, target: torch.Tensor, fourier_weight: float) -> torch.Tensor:

    assert 0.0 <= fourier_weight <= 1.0, "fourier_weight must be between 0 and 1.0"
    assert pred.shape == target.shape, "Prediction and target shapes must match"
    assert pred.dim() == 3, "Pred and target must be 3D tensors of shape (batches, covariates, forecast_horizon)"

    mae_loss = F.l1_loss(pred, target)
    f_loss = fourier_loss(pred, target)
    total_loss = (1 - fourier_weight) * mae_loss + fourier_weight * f_loss
    return total_loss

 
