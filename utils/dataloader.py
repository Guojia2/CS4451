import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
import os


class FreDFDataset(Dataset):
    def __init__(self, root_path,

                 dataset_name,
                 seq_len=96, pred_len=24,
                 split='train',
                 normalize=True):
        # time series dataset for forecasting
        # root_path: dir with csv files.
        # dataset_name: 'etth1', 'ettm1', 'exchange', 'ili', etc (update: there is no etc, we just used these 4 sets)
        # seq_len: lookback window
        # pred_len: forecast horizon
        # split: train, val, or test
        # normalize: z-score using training stats
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.split = split
        self.dataset_name = dataset_name

        # loadd tehj data
        file_path = os.path.join(root_path, f"{dataset_name}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"dataset at {file_path} not found. ")

        # dataset-specific loading
        if dataset_name in ['Exchange', 'exchange']:
            data = self._load_exchange_data(file_path)
        elif dataset_name in ['ILI', 'ili', 'ILINet']:
            data = self._load_ili_data(file_path)
        else:
            #
            data = self._load_standard_data(file_path)

        # get number of features (aka cplumns. )
        self.num_features = data.shape[1]

        # define split boundaries based on dataset type
        train_end, val_end, test_end = self._get_split_boundaries(dataset_name, len(data))

        # normalize using training set stats only
        if normalize:
            train_data = data[:train_end]
            self.scaler = StandardScaler()
            self.scaler.fit(train_data)  # fit on train only (i love avoiding test set leakage i love avoiding test set leakage i love avoiding-)
            data = self.scaler.transform(data)  # transform all

        # extract the train/test/validation split
        if split == 'train':
            self.data = data[:train_end]
        elif split == 'val':
            # start earlier by seq_len to have valid input sequences
            self.data = data[train_end - seq_len:val_end]
        elif split == 'test':
            self.data = data[val_end - seq_len:test_end]
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        # create sliding windows with stride=1
        num_samples = len(self.data) - seq_len - pred_len + 1
        if num_samples <= 0:
            raise ValueError(f"Insufficient data for split '{split}': need at least {seq_len + pred_len} samples")
        
        self.indices = np.arange(num_samples)

    def _load_standard_data(self, file_path):
        # load standard csv. It's ett.
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
        # remove timestamp column.
        if any('date' in c.lower() for c in cols):
            df = df.select(cols[1:])
        df = df.collect()
        return df.to_numpy().astype(np.float64)

    def _load_exchange_data(self, file_path):
        # load foreign exchange rates dataset
        df = pl.scan_csv(file_path)
        cols = df.columns
        # remove date column  on hte far left
        df = df.select(cols[1:])
        df = df.collect()

        # remove the ND values and replace them using forward-filling.
        df = df.with_columns(
            [
                pl.when(pl.col(col) == "ND").then(None).otherwise(pl.col(col)).alias(col)
                for col in df.columns
            ]
        )
        df = df.fill_null(strategy="forward")
        return df.to_numpy().astype(np.float64)

    def _load_ili_data(self, file_path):
        # load ilinet  dataset
        # skip heade rrows
        # read all columns as strings first to handle 'X' values
        df = pl.read_csv(file_path, skip_rows=1, infer_schema_length=0)
        
        # select numeric columns for forecasting
        # drop categorical columns.
        # use weighted ili and age-specific counts
        numeric_cols = ['% WEIGHTED ILI', '%UNWEIGHTED ILI', 'AGE 0-4', 
                       'AGE 5-24', 'AGE 25-49', 'AGE 25-64', 'AGE 50-64', 
                       'AGE 65', 'ILITOTAL', 'TOTAL PATIENTS']
        
        # filter to national level data only for consistency
        df = df.filter(pl.col('REGION TYPE') == 'National')
        
        # handle 'X' values . replace with column mean
        data_list = []
        for col in numeric_cols:
            if col in df.columns:
                # my ide is telling me that there is a more optimal way to do this.
                #my ide does not realize that i am not being paid enough to fix that
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
            # 15 min data: 12-4-8 months
            train_end = 12 * 30 * 96  # 34,560
            val_end = train_end + 4 * 30 * 96  # 46,080
            test_end = val_end + 8 * 30 * 96  # 69,120
        elif dataset_name in ['Exchange', 'exchange']:
            # daily singapre china exhcnag erate data: 5218ish samples (20 years)
            # make the splits
            train_end = int(total_length * 0.7)  #
            val_end = int(total_length * 0.8)    #
            test_end = total_length              #
        elif dataset_name in ['ILI', 'ili', 'ILINet']:
            # weekly data: ~1469 samples
            # use 70-10-20 split


            train_end = int(total_length * 0.7)

            #
            val_end = int(total_length * 0.8)

            test_end = total_length
        else:
            # default to 70-10-20 split bc it seems reasonable enough.
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



def get_train_dataloader(lookback_window, forecast_horizon, batch_size,
                         dataset_name="ETTh1", root_path="./temp/",
                         num_workers=0, shuffle=True):
    train_dataset = FreDFDataset(
        root_path=root_path,
        dataset_name=dataset_name,

        seq_len=lookback_window,

        # look guys i can insert comments into function calls
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




