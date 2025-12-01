# test TSMixer integration with FREDF
import sys
from pathlib import Path

# add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from model import FredF


def test_tsmixer_instantiation():
    print("Test 1: TSMixer instantiation through FredF")
    
    model = FredF(
        covariates=7,
        lookback_window=96,
        forecast_horizon=96,
        backbone='tsmixer',
        e_layers=4,
        d_ff=256,
        dropout=0.1
    )
    
    print(f"  Model created with backbone: {model.backbone_type}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def test_tsmixer_forward():
    print("\nTest 2: TSMixer forward pass")
    
    model = FredF(
        covariates=7,
        lookback_window=96,
        forecast_horizon=96,
        backbone='tsmixer',
        e_layers=4,
        d_ff=256,
        dropout=0.1
    )
    
    # create dummy input: [batch, covariates, lookback]
    batch_size = 4
    x = torch.randn(batch_size, 7, 96)
    
    print(f"  Input shape: {x.shape}")
    
    # forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"  Output shape: {output.shape}")
    assert output.shape == (batch_size, 7, 96), f"Expected shape (4, 7, 96), got {output.shape}"
    print("  Output shape is correct")


def test_itransformer_backward_compatibility():
    print("\nTest 3: itransformer backward compatibility")
    
    model = FredF(
        covariates=7,
        lookback_window=96,
        forecast_horizon=96,
        backbone='itransformer',
        d_model=128,
        n_heads=8,
        e_layers=2,
        d_ff=128,
        dropout=0.1
    )
    
    print(f"  Model created with backbone: {model.backbone_type}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    x = torch.randn(4, 7, 96)
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == (4, 7, 96), f"Expected shape (4, 7, 96), got {output.shape}"
    print("  iTransformer still works correctly")


def test_parameter_comparison():
    print("\nTest 4: Parameter count comparison")
    
    config = {
        'covariates': 7,
        'lookback_window': 96,
        'forecast_horizon': 96,
        'e_layers': 4,
        'd_ff': 256,
        'dropout': 0.1
    }
    
    tsmixer = FredF(**config, backbone='tsmixer')
    itransformer = FredF(**config, backbone='itransformer', d_model=128, n_heads=8)
    
    tsmixer_params = sum(p.numel() for p in tsmixer.parameters())
    itransformer_params = sum(p.numel() for p in itransformer.parameters())
    
    print(f"  TSMixer parameters: {tsmixer_params:,}")
    print(f"  iTransformer parameters: {itransformer_params:,}")
    print(f"  Ratio (iTransformer/TSMixer): {itransformer_params/tsmixer_params:.2f}x")


def test_gradient_flow():
    print("\nTest 5: Gradient flow through TSMixer")
    
    model = FredF(
        covariates=7,
        lookback_window=96,
        forecast_horizon=96,
        backbone='tsmixer',
        e_layers=2,
        d_ff=128,
        dropout=0.1
    )
    
    # Forward pass
    x = torch.randn(2, 7, 96, requires_grad=True)
    output = model(x)
    
    # compute loss and backward
    loss = output.sum()
    loss.backward()
    
    # check gradients
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "No gradients computed!"
    print("  Gradients flow properly through the model")


def test_different_shapes():
    print("\nTest 6: Different input/output shapes")
    
    test_cases = [
        (96, 96, 7),
        (96, 192, 7),
        (96, 336, 7),
        (512, 96, 21),  # Long input like paper
        (36, 24, 10),   # ILI dataset
    ]
    
    for seq_len, pred_len, features in test_cases:
        model = FredF(
            covariates=features,
            lookback_window=seq_len,
            forecast_horizon=pred_len,
            backbone='tsmixer',
            e_layers=2,
            d_ff=128,
            dropout=0.1
        )
        
        x = torch.randn(2, features, seq_len)
        with torch.no_grad():
            output = model(x)
        
        expected_shape = (2, features, pred_len)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"  Shape test passed: [{seq_len} to {pred_len}, {features} features]")


def main():
    print("Testing TSMixer Integration with FREDF")
    print()
    
    try:
        test_tsmixer_instantiation()
        test_tsmixer_forward()
        test_itransformer_backward_compatibility()
        test_parameter_comparison()
        test_gradient_flow()
        test_different_shapes()
        
        print("\nAll tests passed!")
        print("TSMixer is integrated with FREDF.")
        print("Train with: python main.py --backbone [tsmixer|itransformer]")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
