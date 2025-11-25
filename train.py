import torch
from colorama import Fore
from model import FredF
from utils.dataloader import get_train_dataloader, get_val_dataloader
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import Tensor
from utils.losses import fourier_mse_loss

"""
Train a full epoch of the FredF model
Args:
    model (FredF): The  FredF model instance to train.
    dataloader (DataLoader): DataLoader providing training data
    optimizer (Optimizer): Optimizer for updating parameters.
    loss_fn (callable) Loss function to compute the training loss.
    device (torch.device): Device to run the training on (CPU, GPU).
"""
def train_epoch(
    model: FredF,
    dataloader: DataLoader,
    optimizer: Optimizer,
    loss_fn: Callable[[Tensor, Tensor, float], Tensor],
    device: torch.device,
    fourier_weight: float
) -> float:
    
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        # x shape expected: (batches, covariates, lookback)
        optimizer.zero_grad()
        pred = model(x)            
        assert pred.shape == y.shape, \
            f"Prediction shape {pred.shape} doesn't match target shape {y.shape}"
        loss = loss_fn(pred, y, fourier_weight=fourier_weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)

"""
Evaluate the FredF model for one epoch.
Args:
    model (FredF): The FredF model instance to evaluate.
    dataloader (DataLoader): DataLoader providing evaluation data
    loss_fn (callable): Loss function for compute the evaluation loss.
    device (torch.device): Device to run the evaluation on (CPU or GPU).
"""
def eval_epoch(
    model: FredF,
    dataloader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    device: torch.device,
    fourier_weight: float
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            assert pred.shape == y.shape, \
                f"Prediction shape {pred.shape} does not match target shape {y.shape}"
            loss = loss_fn(pred, y, fourier_weight=fourier_weight)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)

"""
Main training loop for the FredF model
"""
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    if device.type == "cpu":
        print(Fore.YELLOW + "Warning: You're on CPU; training will be super slow." + Fore.RESET)

    lookback_window = 10
    forecast_horizon = 5
    covariates = 3

    # TODO DATA ENGINEER set batch size, implement dataloaders in dataloader.py
    train_loader = get_train_dataloader(lookback_window, forecast_horizon, batch_size=16)
    val_loader = get_val_dataloader(lookback_window, forecast_horizon, batch_size=16)
    
    model = FredF(lookback_window=lookback_window, forecast_horizon=forecast_horizon, covariates=covariates).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = fourier_mse_loss
    fourier_weight = 0.5  # TODO DATA ENGINEER tune this hyperparameter 

    epochs = 10 # TODO DATA ENGINEER adjust number of epochs as needed
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, fourier_weight=fourier_weight)
        val_loss = eval_epoch(model, val_loader, loss_fn, device, fourier_weight=fourier_weight) 
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

if __name__ == "__main__":
    main()
