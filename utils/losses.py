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

def fourier_mse_loss(pred: torch.Tensor, target: torch.Tensor, fourier_weight: float) -> torch.Tensor:
    """
    Compute weighted Fourier + MSE  Loss between predictions & targets

    Args:

        pred (Tensor): Predicted vals of shape (batches, covariates, forecast_horizon).
        target (Tensor):  True vals of shape (batches, covariates, forecast_horizon)
        fourier_weight (float): Weighting factor for the Fourier loss component (0 to 1.0)

    Returns:
        Tensor: Discrete Fourier MSE Loss

    """
    assert 0.0 <= fourier_weight <= 1.0, "fourier_weight must be between 0 and 1.0"
    assert pred.shape == target.shape, "Prediction and target shapes must match"
    assert pred.dim() == 3, "Pred and target must be 3D tensors of shape (batches, covariates, forecast_horizon)"

    mse_loss = F.mse_loss(pred, target)
    f_loss = fourier_loss(pred, target)
    total_loss = (1 - fourier_weight) * mse_loss + fourier_weight * f_loss
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

 
