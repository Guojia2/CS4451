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
├── train.py
├── model.py # old code for FreDF model
├── models
│   ├── iTransformer.py
│   ├── autoFormer.py
│   ├── scaleFormer.py
│   ├── patchTST.py
│   ├── timeMixer.py
│   └── DLinear.py 
├── utils 
│   ├── layers.py # nn.Module definitions for FFT layers 
│   ├── inverse-fft.py # convert output labels back from frequency domain
│   ├── dataloader.py # dataloader definitions 
│   ├── test_dataloader.py # used for testing dataloader class
│   └── losses.py # definitions for loss functions 
├── pyproject.toml
├── README.md
└── uv.lock 
```
