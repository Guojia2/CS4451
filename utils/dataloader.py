import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
import os


class FreDFDataset(Dataset):
    def __init__(self, root_path, dataset_name,
                 seq_len=96, pred_len=24,
                 split='train', split_ratio=(0.7, 0.2, 0.1),
                 normalize=True):
        """
        root_path: directory containing csv dataset files (e.g. './data/')
        dataset_name: e.g. 'ETTm1', 'ETTh1', 'ECL', 'Traffic', etc.
        seq_len: input sequence length
        pred_len: forecast horizon
        split: 'train', 'val', 'test'
        split_ratio: fractional splits by time
        normalize: whether to z-score features
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.split = split

        # find the dataset files
        file_path = os.path.join(root_path, f"{dataset_name}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset {file_path} not found!")

        # --- Lazy read with Polars ---
        # I failed an interview because I didn't know that Polars allows lazy execution.
        df = pl.scan_csv(file_path)

        # Detect & remove timestamp column if present
        cols = df.columns
        if any('date' in c.lower() or 'time' in c.lower() for c in cols):
            df = df.select(cols[1:])  # skip first column

        # Collect after scanning (efficient)
        df = df.collect()

        # Convert to numpy

        data = df.to_numpy().astype(np.float32) # TODO: change to float16 if the values don't requure 32 bits

        #  Normalization
        if normalize:
            self.scaler = StandardScaler()
            data = self.scaler.fit_transform(data)

        # --- Split by time
        n = len(data)
        n_train = int(n * split_ratio[0])
        n_val = int(n * split_ratio[1])

        if split == 'train':
            self.data = data[:n_train]
        elif split == 'val':
            self.data = data[n_train:n_train + n_val]
        elif split == 'test':
            self.data = data[n_train + n_val:]
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        # Precompute valid start indices
        self.indices = np.arange(len(self.data) - seq_len - pred_len + 1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        seq_x = self.data[start:start + self.seq_len]
        seq_y = self.data[start + self.seq_len:start + self.seq_len + self.pred_len]

        # iTransformer expects [batch, num_features, seq_len]
        seq_x = torch.tensor(seq_x.T, dtype=torch.float32)
        seq_y = torch.tensor(seq_y.T, dtype=torch.float32)

        return seq_x, seq_y


# Example usage.
if __name__ == "__main__":
    dataset = FreDFDataset("./data/", "ETTh1", seq_len=96, pred_len=24, split='train')
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)

    for x, y in loader:
        print("x:", x.shape, "y:", y.shape)
        break


def get_train_dataloader(lookback_window, forecast_horizon, batch_size,
                         dataset_name="ETTh1", root_path="./data/",
                         num_workers=2, shuffle=True):
    # TODO DATA ENGINEER Create and return a DataLoader instance
    """
    Returns a DataLoader for the training split of the dataset.
    """
    train_dataset = FreDFDataset(
        root_path=root_path,
        dataset_name=dataset_name,
        seq_len=lookback_window,
        pred_len=forecast_horizon,
        split='train'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True
    )

    return train_loader



def get_val_dataloader(lookback_window, forecast_horizon, batch_size,
                       dataset_name="ETTh1", root_path="./data/",
                       num_workers=2, shuffle=False):
    # TODO DATA ENGINEER Create and return a DataLoader instance

    """
       Returns a DataLoader for the validation split of the dataset.
       """
    val_dataset = FreDFDataset(
        root_path=root_path,
        dataset_name=dataset_name,
        seq_len=lookback_window,
        pred_len=forecast_horizon,
        split='val'
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False
    )

    return val_loader



