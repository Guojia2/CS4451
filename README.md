
## Welcome to the branch test/dataloader

### Here are some issues and notes:

#### main.py:
#### dataloader.py:

- This thing does NOT handle the "Exchange" dataset currently,because in _load_exchange_data, we get an error from trying to convert 'ND' to a string. THe ND value appears to mean "No data", so a null value. I'm going to try to fix that.

#### train/train.py:

- This file does not accept any dataset argumebts except for ETTh1, ETTm1, Exchange, and ILI. I'm not sure if this is intentionally. Perhaps we aren't using datasets besides thhose?
- update: based on the contents of the dataset_configs dictionary in train.py, it seems we are only using those 4 datsets:
  -     dataset_configs = {
          'ETTh1': {'seq_len': 96, 'pred_len': 96, 'features': 7, 'd_ff': 128},
          'ETTm1': {'seq_len': 96, 'pred_len': 96, 'features': 7, 'd_ff': 128},
          'Exchange': {'seq_len': 96, 'pred_len': 96, 'features': 2, 'd_ff': 128},
          'ILI': {'seq_len': 36, 'pred_len': 24, 'features': 10, 'd_ff': 64},
      }
### dataset notes:

####  Exchange dataset
- I'm not sure where Calvin got the Exchange dataset he sent me. When I follow the links he sent me and go to https://github.com/laiguokun/multivariate-time-series-data, I end up with a .txt file which is formatted differently and causes our training pipeline to throw an error.
- Exchange contains some values entered as ND, presumably meaning "no data". I'm unsure if we should delete these rows or impute.
    - imputation would probably be one of the following:
      - linear interpolation (assume lienar relationship between prev and future values)
      - Backward fill (repalce ND value with the value preceeding it)
      - Forward fill (Replace ND value witht eh value succeeding it)
    - will have to ask teammates (Calvin, probably) for how to proceed.

####  ETTh1 and ETTm1

- Float32 is definitely insuffcient to maintaint the precision for ETTh1.csv anf for ETTm1. We need to move up to float64, maybe even float128.
#### ILI
- ILI also may need float64, similar to ETTh1 and ETTm1, I'm not sure. 
- ILI contains stretches of 0 values. There are entries like National,X,1998,24,0,0,0,0,0,0,0,0,0,0,0. THis surely cannot be correct, right? 
  - Perhaps must consider deleting these entries? I'm not sure if that is allowed for time series data. Further inquiry required.




# FreDF Implementation Overview 

This repo contains the FreDF paper implementation for improving upon Direct Forecasting (DF) using Fast Fourier Transform (FFT) on the inputs and labels during the training of various models, including:

| Model Name | Architecture |
| ---------- | ------------ |
| iTransformer | transformer |
| ScaleFormer | transformer |
| AutoFormer | transformer |
| PatchTST | transformer |
| TimeMixer | MLP |
| DLinear | MLP |

File Tree: 
```
├── main.py 
├── train.py # old code for training 
├── model.py # old code for FreDF model
├── models # folder for all the models 
│   ├── __init__.py  
│   ├── iTransformer 
│   │   ├── __init__.py
│   │   ├── Embed.py
│   │   ├── model.py
│   │   ├── SelfAttention_Family.py
│   │   ├── Transformer_EncDec.py
│   ├── TSMixer.py 
├── trains # contains code for training the models 
│   ├── train.py # example train file copied from old code  
│   ├── exampleitransformer.py  # example usage for iTransformer 
│   ├── dummyitransformers.py # dummy data with iTransformer 
├── utils # utils folder for functions and modules 
│   ├── fft-layer.py # nn.Module definition for FFT layer
│   ├── inverse-fft.py # convert output labels back from frequency domain
│   ├── dataloader.py # dataloader definitions 
│   ├── test_dataloader.py # used for testing dataloader class
│   └── losses.py # definitions for loss functions 
├── pyproject.toml
├── README.md
└── uv.lock 
```

## Running iTransformer

Activate the uv venv, sync deps with `uv sync`, then run in the project root: `python -m trains.exampleitransformer.py`
