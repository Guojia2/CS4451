import sys
from pathlib import Path

# add parent directory to path so we can import model and utils
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from colorama import Fore
from model import FredF
from utils.dataloader import get_train_dataloader, get_val_dataloader, get_test_dataloader
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import Tensor
from utils.losses import fredf_loss


def train_epoch(
    model: FredF,
    dataloader: DataLoader,
    optimizer: Optimizer,
    loss_fn: Callable[[Tensor, Tensor, float], Tensor],
    device: torch.device,
    fourier_weight: float
) -> float:
    # train one epoch of fredf
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


def eval_epoch(
    model: FredF,
    dataloader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor, float], Tensor],
    device: torch.device,
    fourier_weight: float
) -> float:
    # eval fredf for one epoch
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


def compute_metrics(model: FredF, dataloader: DataLoader, device: torch.device):
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            mse = torch.nn.functional.mse_loss(pred, y, reduction='sum')
            mae = torch.nn.functional.l1_loss(pred, y, reduction='sum')
            total_mse += mse.item()
            total_mae += mae.item()
    
    num_samples = len(dataloader.dataset) * y.shape[1] * y.shape[2]
    return total_mse / num_samples, total_mae / num_samples


def main() -> None:
    # main training loop for fredf
    # supports etth1, exchange, ili datasets
    # can choose between itransformer and tsmixer backbone
    import argparse
    
    parser = argparse.ArgumentParser(description='FreDF Training')
    parser.add_argument('--dataset', type=str, default='ETTh1', 
                       choices=['ETTh1', 'ETTm1', 'Exchange', 'ILI'],
                       help='Dataset name')
    parser.add_argument('--backbone', type=str, default='itransformer',
                       choices=['itransformer', 'tsmixer'],
                       help='Backbone architecture (itransformer or tsmixer)')
    parser.add_argument('--seq_len', type=int, default=None, 
                       help='Input sequence length (auto-set based on dataset if not provided)')
    parser.add_argument('--pred_len', type=int, default=None,
                       help='Forecast horizon (auto-set based on dataset if not provided)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lambda_freq', type=float, default=0.2, 
                       help='Weight for frequency loss (lambda)')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # interpretability options
    parser.add_argument('--run_interp', action='store_true',
                       help='run interpretability analysis after training')
    parser.add_argument('--interp_methods', type=str, default='ig,attention,masking',
                       help='interpretability methods: ig, attention, masking (comma-separated)')
    parser.add_argument('--interp_samples', type=int, default=3,
                       help='number of test samples for interpretability')
    parser.add_argument('--interp_indices', type=str, default=None,
                       help='specific test indices for interpretability (e.g. "0,5,10")')
    parser.add_argument('--interp_output_dir', type=str, default='./interpretability_results',
                       help='output directory for interpretability visualizations')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    if device.type == "cpu":
        print(Fore.YELLOW + "warning: using CPU, training will be slow" + Fore.RESET)

    # dataset configs
    dataset_configs = {
        'ETTh1': {'seq_len': 96, 'pred_len': 96, 'features': 7, 'd_ff': 128},
        'ETTm1': {'seq_len': 96, 'pred_len': 96, 'features': 7, 'd_ff': 128},
        'Exchange': {'seq_len': 96, 'pred_len': 96, 'features': 2, 'd_ff': 128},
        'ILI': {'seq_len': 36, 'pred_len': 24, 'features': 10, 'd_ff': 64},
    }
    
    config = dataset_configs[args.dataset]
    
    # use provided values or defaults from config
    lookback_window = args.seq_len if args.seq_len else config['seq_len']
    forecast_horizon = args.pred_len if args.pred_len else config['pred_len']
    covariates = config['features']
    d_ff = config['d_ff']
    
    print(f"\nTraining FreDF + {args.backbone.upper()} on {args.dataset}")
    print(f"Configuration:")
    print(f"Dataset: {args.dataset}")
    print(f"Backbone: {args.backbone}")
    print(f"Input length (seq_len): {lookback_window}")
    print(f"Forecast horizon (pred_len): {forecast_horizon}")
    print(f"Features: {covariates}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Lambda (frequency weight): {args.lambda_freq}")
    print(f"Model: d_model={args.d_model}, n_heads={args.n_heads}, e_layers={args.e_layers}, d_ff={d_ff}")
    print()
    
    # data loading
    print(f"Loading {args.dataset} dataset...")
    train_loader = get_train_dataloader(
        lookback_window, forecast_horizon, batch_size=args.batch_size,
        dataset_name=args.dataset, root_path="./temp/", num_workers=0
    )
    val_loader = get_val_dataloader(
        lookback_window, forecast_horizon, batch_size=args.batch_size,
        dataset_name=args.dataset, root_path="./temp/", num_workers=0
    )
    test_loader = get_test_dataloader(
        lookback_window, forecast_horizon, batch_size=args.batch_size,
        dataset_name=args.dataset, root_path="./temp/", num_workers=0
    )
    
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
    
    # model init
    model = FredF(
        covariates=covariates,
        lookback_window=lookback_window,
        forecast_horizon=forecast_horizon,
        backbone=args.backbone,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=d_ff,
        dropout=args.dropout
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = fredf_loss  # fredf paper loss: λ*L_freq + (1-λ)*L_temp
    
    print(f"\nTraining with fourier_weight (λ) = {args.lambda_freq}")
    print(f"Loss: {args.lambda_freq} * L_freq + {1-args.lambda_freq} * L_temp\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, fourier_weight=args.lambda_freq)
        val_loss = eval_epoch(model, val_loader, loss_fn, device, fourier_weight=args.lambda_freq)
        
        # compute metrics
        val_mse, val_mae = compute_metrics(model, val_loader, device)
        
        print(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_mse={val_mse:.4f}, val_mae={val_mae:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'./checkpoints/best_model_{args.dataset}_{args.backbone}.pt')
            print(f"  -> Best model saved (val_loss={val_loss:.4f})")
    
    # final test eval
    print("\nFinal Test Evaluation:")
    model.load_state_dict(torch.load(f'./checkpoints/best_model_{args.dataset}_{args.backbone}.pt'))
    test_mse, test_mae = compute_metrics(model, test_loader, device)
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # optionally run interpretability
    if args.run_interp:
        print("\n" + "="*50)
        print("running interpretability analysis...")
        print("="*50)
        from utils.interpret import run_interpretability
        run_interpretability(model, test_loader, device, args)


if __name__ == "__main__":
    import os
    os.makedirs('./checkpoints', exist_ok=True)
    main()
