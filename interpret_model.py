"""
script for running interpretability analysis on trained models
supports integrated gradients, attention visualization, and patch masking
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import argparse
from model import FredF
from utils.dataloader import get_test_dataloader
from utils.interpret import run_interpretability


def main():
    parser = argparse.ArgumentParser(description='FreDF Model Interpretability')
    
    # model and dataset args
    parser.add_argument('--dataset', type=str, default='ETTh1',
                       choices=['ETTh1', 'ETTm1', 'Exchange', 'ILI'],
                       help='which dataset')
    parser.add_argument('--backbone', type=str, default='itransformer',
                       choices=['itransformer', 'tsmixer'],
                       help='model backbone')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='path to model checkpoint (default: auto-detect from dataset/backbone)')
    
    # model architecture (should match training)
    parser.add_argument('--seq_len', type=int, default=None,
                       help='input sequence length (auto from dataset if not provided)')
    parser.add_argument('--pred_len', type=int, default=None,
                       help='forecast horizon (auto from dataset if not provided)')
    parser.add_argument('--d_model', type=int, default=128,
                       help='model dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                       help='num attention heads')
    parser.add_argument('--e_layers', type=int, default=2,
                       help='num encoder layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='dropout rate')
    
    # interpretability settings
    parser.add_argument('--interp_methods', type=str, default='ig,attention,masking',
                       help='comma-separated list: ig, attention, masking')
    parser.add_argument('--interp_samples', type=int, default=3,
                       help='number of random test samples to analyze')
    parser.add_argument('--interp_indices', type=str, default=None,
                       help='specific test indices (comma-separated, e.g. "0,5,10")')
    parser.add_argument('--interp_horizons', type=str, default=None,
                       help='which horizon steps to analyze (comma-separated, default: first,mid,last)')
    parser.add_argument('--interp_output_dir', type=str, default='./interpretability_results',
                       help='where to save visualizations')
    parser.add_argument('--target_variate', type=int, default=None,
                       help='which output variate to target for importance (0-indexed, default: all)')
    
    # method-specific params
    parser.add_argument('--ig_steps', type=int, default=50,
                       help='number of interpolation steps for integrated gradients')
    parser.add_argument('--patch_size', type=int, default=8,
                       help='patch size for masking analysis')
    
    # data loading
    parser.add_argument('--batch_size', type=int, default=32,
                       help='batch size for data loading')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    
    # dataset configs (should match training)
    dataset_configs = {
        'ETTh1': {'seq_len': 96, 'pred_len': 96, 'features': 7, 'd_ff': 128},
        'ETTm1': {'seq_len': 96, 'pred_len': 96, 'features': 7, 'd_ff': 128},
        'Exchange': {'seq_len': 96, 'pred_len': 96, 'features': 2, 'd_ff': 128},
        'ILI': {'seq_len': 36, 'pred_len': 24, 'features': 10, 'd_ff': 64},
    }
    
    config = dataset_configs[args.dataset]
    lookback_window = args.seq_len if args.seq_len else config['seq_len']
    forecast_horizon = args.pred_len if args.pred_len else config['pred_len']
    covariates = config['features']
    d_ff = config['d_ff']
    
    print(f"\nloading model:")
    print(f"  dataset: {args.dataset}")
    print(f"  backbone: {args.backbone}")
    print(f"  seq_len: {lookback_window}, pred_len: {forecast_horizon}")
    print(f"  features: {covariates}")
    
    # build model
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
    
    # load checkpoint
    if args.checkpoint is None:
        checkpoint_path = f'./checkpoints/best_model_{args.dataset}_{args.backbone}.pt'
    else:
        checkpoint_path = args.checkpoint
    
    print(f"\nloading checkpoint: {checkpoint_path}")
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("  checkpoint loaded successfully")
    except FileNotFoundError:
        print(f"error: checkpoint not found at {checkpoint_path}")
        return
    
    # load test data
    print(f"\nloading test data...")
    test_loader = get_test_dataloader(
        lookback_window, forecast_horizon,
        batch_size=args.batch_size,
        dataset_name=args.dataset,
        root_path='./temp/',
        num_workers=0
    )
    print(f"  test samples: {len(test_loader.dataset)}")
    
    # parse methods
    args.interp_methods = [m.strip() for m in args.interp_methods.split(',')]
    print(f"\ninterpretability methods: {args.interp_methods}")
    
    # run interpretability
    run_interpretability(model, test_loader, device, args)
    
    print("\n end of report.")


if __name__ == "__main__":
    main()
