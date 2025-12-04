
## Welcome to the branch test/dataloader

### Here are some issues and notes:

#### interpret.py:
- the masking function does NOT work for the itransformer backbone. Or for the TSMixer backbone.
- Attention maps work for iTransformer just fine
- IG does not work for itrasnformer or TSMixer backbones. I think there is some argument parsing issue or issue with setting default values
#### dataloader.py:

- This thing does NOT handle the "Exchange" dataset currently,because in _load_exchange_data, we get an error from trying to convert 'ND' to a string. THe ND value appears to mean "No data", so a null value. I'm going to try to fix that.
- update Dec 3 2025: Fixed it. Now handles teh nulls and forward-fills them
#### train/train.py:

### dataset notes:

####  Exchange dataset
- I'm not sure where Calvin got the Exchange dataset he sent me. When I follow the links he sent me and go to https://github.com/laiguokun/multivariate-time-series-data, I end up with a .txt file which is formatted differently and causes our training pipeline to throw an error.
- update Dec 3 2025: Calvin says he got a clean version of it from somewhere. Also, I filled nulls with foward fill to make the dataloader handle it
####  ETTh1 and ETTm1

- Float32 is definitely insuffcient to maintaint the precision for ETTh1.csv anf for ETTm1. We need to move up to float64, maybe even float128.
- update: 12/3/2025: made teh changes to float64. Dataloders and model constructors had to be SLIGHTLY modified but that it ok
#### ILI
- ILI also may need float64, similar to ETTh1 and ETTm1, I'm not sure. 
- ILI contains stretches of 0 values. There are entries like National,X,1998,24,0,0,0,0,0,0,0,0,0,0,0. THis surely cannot be correct, right? 
  - Perhaps must consider deleting these entries? I'm not sure if that is allowed for time series data. Further inquiry required.
- update 12/3/2025: made the change to float64. 0 values are being left as-is because not enough time to deal with taht rn



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
