import torch
import numpy as np
import matplotlib.pyplot as plt
from dataloader import get_train_dataloader, get_val_dataloader

def summarize_batch(x, y):
    # print shape and stats for a sample batch
    print("\nData Loader Test Summary")
    print(f"Input  batch shape : {x.shape}")   # [B, num_features, seq_len]
    print(f"Target batch shape : {y.shape}")   # [B, num_features, pred_len]

    # check normalization
    mean = x.mean().item()
    std = x.std().item()
    print(f"Input mean ≈ {mean:.4f}, std ≈ {std:.4f}")
    print(f"Target mean ≈ {y.mean().item():.4f}, std ≈ {y.std().item():.4f}")

def plot_sample(x, y, feature_idx=0):
    # plot one feature's input and target segment
    x_np = x[0, feature_idx].cpu().numpy()
    y_np = y[0, feature_idx].cpu().numpy()
    t_in = np.arange(len(x_np))
    t_out = np.arange(len(x_np), len(x_np) + len(y_np))

    plt.figure(figsize=(8, 3))
    plt.plot(t_in, x_np, label="Input (past)")
    plt.plot(t_out, y_np, label="Target (future)", color="orange")
    plt.title(f"Feature {feature_idx} — One Forecast Window")
    plt.xlabel("Timestep")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    lookback = 96
    horizon = 24
    batch_size = 32
    dataset_name = "ETTm1"   # make sure ./data/ETTm1.csv exists

    print("Loading training DataLoader...")
    train_loader = get_train_dataloader(lookback, horizon, batch_size, dataset_name=dataset_name)

    # grab one batch
    x, y = next(iter(train_loader))

    summarize_batch(x, y)
    plot_sample(x, y)

    print(" DataLoader test completed successfully.")

if __name__ == "__main__":
    main()
