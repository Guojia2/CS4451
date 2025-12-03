import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
import os


class FreDFDataset(Dataset):
    def __init__(self, root_path, dataset_name,
                 seq_len=96, pred_len=24,
                 split='train',
                 normalize=True):
        # time series dataset for forecasting
        # root_path: dir with csv files (e.g. './temp/')
        # dataset_name: 'etth1', 'ettm1', 'exchange', 'ili', etc
        # seq_len: lookback window
        # pred_len: forecast horizon
        # split: 'train', 'val', or 'test'
        # normalize: z-score using training stats
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.split = split
        self.dataset_name = dataset_name

        # load and preprocess dataset
        file_path = os.path.join(root_path, f"{dataset_name}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset {file_path} not found!")

        # dataset-specific loading
        if dataset_name in ['Exchange', 'exchange']:
            data = self._load_exchange_data(file_path)
        elif dataset_name in ['ILI', 'ili', 'ILINet']:
            data = self._load_ili_data(file_path)
        else:
            # standard ett and other datasets
            data = self._load_standard_data(file_path)

        # get number of features
        self.num_features = data.shape[1]

        # define split boundaries based on dataset type
        train_end, val_end, test_end = self._get_split_boundaries(dataset_name, len(data))

        # normalize using training set statistics only
        if normalize:
            train_data = data[:train_end]
            self.scaler = StandardScaler()
            self.scaler.fit(train_data)  # fit on train only
            data = self.scaler.transform(data)  # transform all

        # extract the appropriate split
        if split == 'train':
            self.data = data[:train_end]
        elif split == 'val':
            # start earlier by seq_len to have valid input sequences
            self.data = data[train_end - seq_len:val_end]
        elif split == 'test':
            self.data = data[val_end - seq_len:test_end]
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        # create sliding windows with stride=1 (overlapping)
        num_samples = len(self.data) - seq_len - pred_len + 1
        if num_samples <= 0:
            raise ValueError(f"Insufficient data for split '{split}': need at least {seq_len + pred_len} samples")
        
        self.indices = np.arange(num_samples)

    def _load_standard_data(self, file_path):
        # load standard csv (ett, weather, etc)
        full_schema = {
            "date": pl.String,
            "HUFL": pl.Float64,
            "HULL": pl.Float64,
            "MULL": pl.Float64,
            "MUFL": pl.Float64,
            "LUFL": pl.Float64,
            "LULL": pl.Float64,
            "OT": pl.Float64,
        }

        df = pl.scan_csv(file_path, schema=full_schema)
        cols = df.columns
        # remove timestamp column if present (usually first column)
        if any('date' in c.lower() or 'time' in c.lower() for c in cols):
            df = df.select(cols[1:])
        df = df.collect()
        return df.to_numpy().astype(np.float64)

    def _load_exchange_data(self, file_path):
        # load foreign exchange rates dataset
        df = pl.scan_csv(file_path)
        cols = df.columns
        # remove date column (first column)
        df = df.select(cols[1:])
        df = df.collect()
        df = df.with_columns(
            [
                pl.when(pl.col(col) == "ND").then(None).otherwise(pl.col(col)).alias(col)
                for col in df.columns
            ]
        )
        return df.to_numpy().astype(np.float64) # This line right here is where our training pipleine fails because it cannot handle the ND entries.

    def _load_ili_data(self, file_path):
        # load ilinet (influenza-like illness) dataset
        # ili has 2 header rows - skip the first descriptive row
        # read all columns as strings first to handle 'X' values
        df = pl.read_csv(file_path, skip_rows=1, infer_schema_length=0)
        
        # select numeric columns for forecasting
        # drop categorical columns: REGION TYPE, REGION, YEAR, WEEK
        # use weighted ili and age-specific counts
        numeric_cols = ['% WEIGHTED ILI', '%UNWEIGHTED ILI', 'AGE 0-4', 
                       'AGE 5-24', 'AGE 25-49', 'AGE 25-64', 'AGE 50-64', 
                       'AGE 65', 'ILITOTAL', 'TOTAL PATIENTS']
        
        # filter to national level data only for consistency
        df = df.filter(pl.col('REGION TYPE') == 'National')
        
        # handle 'X' values (missing data) - replace with column mean
        data_list = []
        for col in numeric_cols:
            if col in df.columns:
                series = df[col]
                # convert to string, replace 'X' with None, then to float
                values = []
                for val in series:
                    if val == 'X' or val == 'x':
                        values.append(None)
                    else:
                        try:
                            values.append(float(val))
                        except:
                            values.append(None)
                # fill none with column mean
                valid_values = [v for v in values if v is not None]
                if valid_values:
                    mean_val = np.mean(valid_values)
                    values = [v if v is not None else mean_val for v in values]
                else:
                    values = [0.0] * len(values)
                data_list.append(values)
        
        data = np.array(data_list, dtype=np.float64).T
        return data

    def _get_split_boundaries(self, dataset_name, total_length):
        # get train/val/test split boundaries for each dataset
        if 'ETTh' in dataset_name:
            # hourly data: 12-4-8 months
            train_end = 12 * 30 * 24  # 8,640
            val_end = train_end + 4 * 30 * 24  # 11,520
            test_end = val_end + 8 * 30 * 24  # 17,280
        elif 'ETTm' in dataset_name:
            # 15-min data: 12-4-8 months
            train_end = 12 * 30 * 96  # 34,560
            val_end = train_end + 4 * 30 * 96  # 46,080
            test_end = val_end + 8 * 30 * 96  # 69,120
        elif dataset_name in ['Exchange', 'exchange']:
            # daily forex data: ~5218 samples (20 years)
            # use 70-10-20 split for daily data
            train_end = int(total_length * 0.7)  # ~3,653
            val_end = int(total_length * 0.8)    # ~4,174
            test_end = total_length              # ~5,218
        elif dataset_name in ['ILI', 'ili', 'ILINet']:
            # weekly data: ~1469 samples
            # use 70-10-20 split
            train_end = int(total_length * 0.7)  # ~1,028
            val_end = int(total_length * 0.8)    # ~1,175
            test_end = total_length              # ~1,469
        else:
            # default: 70-10-20 split
            train_end = int(total_length * 0.7)
            val_end = int(total_length * 0.8)
            test_end = total_length
        
        return train_end, val_end, test_end

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # returns one sample with overlapping windows
        # structure: |-- seq_x (input) --|-- seq_y (target) --|
        # stride=1 means consecutive samples overlap by (seq_len-1) timesteps
        start = self.indices[idx]
        seq_x = self.data[start:start + self.seq_len]
        seq_y = self.data[start + self.seq_len:start + self.seq_len + self.pred_len]

        # transpose to [num_features, seq_len] format for fredf/itransformer
        # shape: [time, features] -> [features, time]
        seq_x = torch.tensor(seq_x.T, dtype=torch.float64)
        seq_y = torch.tensor(seq_y.T, dtype=torch.float64)

        return seq_x, seq_y


# example usage and verification
if __name__ == "__main__":
    print("Testing FreDFDataset with multiple datasets...")
    print()
    
    # test etth1
    print("1. ETTh1 Dataset (Hourly Electricity Transformer Temperature)")
    try:
        dataset = FreDFDataset("./temp/", "ETTh1", seq_len=96, pred_len=96, split='train')
        print(f"  Train: {len(dataset)} samples, {dataset.num_features} features")
        val_dataset = FreDFDataset("./temp/", "ETTh1", seq_len=96, pred_len=96, split='val')
        print(f"Val:   {len(val_dataset)} samples")
        test_dataset = FreDFDataset("./temp/", "ETTh1", seq_len=96, pred_len=96, split='test')
        print(f"Test:  {len(test_dataset)} samples")
        
        loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
        x, y = next(iter(loader))
        print(f" Batch shape: x={x.shape}, y={y.shape}")
        print(f"   ETTh1 loaded successfully!")
    except Exception as e:
        print(f"   Error loading ETTh1: {e}")
    
    # test exchange
    print("\n2. Exchange Dataset (Daily Foreign Exchange Rates)")
    try:
        dataset = FreDFDataset("./temp/", "Exchange", seq_len=96, pred_len=96, split='train')
        print(f"Train: {len(dataset)} samples, {dataset.num_features} features")
        val_dataset = FreDFDataset("./temp/", "Exchange", seq_len=96, pred_len=96, split='val')
        print(f"Val:   {len(val_dataset)} samples")
        test_dataset = FreDFDataset("./temp/", "Exchange", seq_len=96, pred_len=96, split='test')
        print(f"Test:  {len(test_dataset)} samples")
        
        loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
        x, y = next(iter(loader))
        print(f"   Batch shape: x={x.shape}, y={y.shape}")
        print(f"  Exchange loaded successfully!")
    except Exception as e:
        print(f" error loading Exchange: {e}")
    
    # test ili
    print("\n3. ILI Dataset (Weekly Influenza-Like Illness)")
    try:
        dataset = FreDFDataset("./temp/", "ILI", seq_len=36, pred_len=24, split='train')
        print(f"Train: {len(dataset)} samples, {dataset.num_features} features")
        val_dataset = FreDFDataset("./temp/", "ILI", seq_len=36, pred_len=24, split='val')
        print(f"Val:   {len(val_dataset)} samples")
        test_dataset = FreDFDataset("./temp/", "ILI", seq_len=36, pred_len=24, split='test')
        print(f"Test:  {len(test_dataset)} samples")
        
        loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
        x, y = next(iter(loader))
        print(f"   Batch shape: x={x.shape}, y={y.shape}")
        print(f"   ILI loaded successfully!")
    except Exception as e:
        print(f"   Error loading ILI: {e}")
    
    print("\nDataset loading complete!!!!")
    print("note: seq_len/pred_len varies by frequency:")
    print("   ETTh1/ETTm1: seq_len=96 (hourly/15-min)")
    print("  Exchange: seq_len=96 (daily)")
    print("  ILI: seq_len=36 (weekly)")



def get_train_dataloader(lookback_window, forecast_horizon, batch_size,
                         dataset_name="ETTh1", root_path="./temp/",
                         num_workers=0, shuffle=True):
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
                       dataset_name="ETTh1", root_path="./temp/",
                       num_workers=0, shuffle=False):
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


def get_test_dataloader(lookback_window, forecast_horizon, batch_size,
                        dataset_name="ETTh1", root_path="./temp/",
                        num_workers=0, shuffle=False):
    test_dataset = FreDFDataset(
        root_path=root_path,
        dataset_name=dataset_name,
        seq_len=lookback_window,
        pred_len=forecast_horizon,
        split='test'
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False
    )

    return test_loader




