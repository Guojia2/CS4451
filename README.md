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
│   ├── autoFormer.py # will turn into folders if needed.
│   ├── scaleFormer.py
│   ├── patchTST.py
│   ├── timeMixer.py
│   └── DLinear.py 
├── trains # contains code for training the models 
│   ├── train.py # example train file copied from old code  
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
