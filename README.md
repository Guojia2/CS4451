# FreDF Implementation Overview 

This repo contains the FreDF paper implementation for improving upon Direct Forecasting (DF) using Fast Fourier Transform (FFT) on the inputs and labels during training and computing the fourier loss in the frequency domain for both the generated sequences and the labels to reduce estimator bias caused by label autocorrelation. This repo benchmarks the performance of FreDF on various datasets, including datasets from the ETT datasets, Exchange and ILI influenza for TSMixer and iTransformer with FreDF implemented.

## Datasets Used:

- [ETTm1 and ETTh1 datasets](https://github.com/zhouhaoyi/ETDataset)
- [Exchange rate dataset](https://github.com/datasets/exchange-rates)
- [ILI influenza dataset](https://github.com/alireza-jafari/ILI-Influenza-Dataset)

## File Tree: 

```
├── models/
│   ├── __pycache__/
│   ├── iTransformer/ # iTransformer definition with FreDF
│   │   ├── __pycache__/
│   │   ├── Embed.py
│   │   ├── SelfAttention_Family.py
│   │   ├── Transformer_EncDec.py
│   │   ├── __init__.py
│   │   ├── masking.py
│   │   └── model.py
│   ├── TSMixer.py # TSMixer definition 
│   └── __init__.py
├── trains/
│   ├── exampleitransformer.py # example iTransformer training code 
│   └── train.py 
├── utils/
│   ├── __pycache__/
│   ├── dataloader.py # dataloader definition
│   ├── interpret.py
│   ├── inverse-fft.py # inverse transform function for converting back from frequency domain 
│   └── losses.py # custom loss function 
├── README.md
├── hyperparam_search.sh # hyperparameter search
├── interpret_model.py
├── main.py
├── model.py # used in hyperparam_search.sh
├── pyproject.toml
└── uv.lock
```

## Running iTransformer

Requirements: `uv` installed from the [Astral](https://docs.astral.sh/uv/getting-started/installation/) docs.

Activate the uv venv, sync deps with `uv sync`, then run in the project root: `python -m trains.exampleitransformer.py`


## Dataset Notes:

###  ETTh1 and ETTm1

- Float32 was insuffcient to maintaint the precision for ETTh1.csv anf for ETTm1, Float64 was used for training instead.

### ILI

- ILI contains stretches of 0 values. There are entries like National,X,1998,24,0,0,0,0,0,0,0,0,0,0,0, which we processed before training.
- Float64 was used for loading this dataset during training.

# References  

- [FreDF](https://arxiv.org/abs/2402.02399) 
- [TSMixer architecture](https://arxiv.org/abs/2303.06053)
- [iTransformer architecture](https://arxiv.org/abs/2310.06625)

