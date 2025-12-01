import inspect
from typing import Any, Dict

import torch
from torch import nn

from models import TSMixer
from models.iTransformer.model import Model as iTransformer


class ITransformerConfig:
    def __init__(self, seq_len, pred_len, enc_in, dec_in, d_model=128, n_heads=8, 
                 e_layers=2, d_ff=128, dropout=0.1, activation='gelu', 
                 output_attention=False, embed='timeF', freq='h', use_norm=True):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        self.output_attention = output_attention
        self.embed = embed
        self.freq = freq
        self.use_norm = use_norm
        self.factor = 3  # attention factor
        self.class_strategy = 'projection'


class ITransformerWrapper(nn.Module):
    # handles shape conversions for fredf
    # input: (batch, covariates, lookback_window)
    # output: (batch, covariates, forecast_horizon)
    def __init__(self, seq_len: int, pred_len: int, enc_in: int, dec_in: int = None,
                 d_model: int = 128, n_heads: int = 8, e_layers: int = 2, 
                 d_ff: int = 128, dropout: float = 0.1, **kwargs):
        super().__init__()
        dec_in = dec_in or enc_in
        
        config = ITransformerConfig(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            dec_in=dec_in,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout
        )
        
        self.model = iTransformer(config)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, covariates, lookback) -> need (batch, lookback, covariates)
        b, c, t = x.shape
        assert c == self.enc_in and t == self.seq_len, \
            f"Expected shape [B, {self.enc_in}, {self.seq_len}], got {x.shape}"
        
        # transpose to [B, L, C] for itransformer
        x = x.transpose(1, 2)
        
        # itransformer doesnt use time marks in our simple case
        x_mark_enc = None
        x_dec = None
        x_mark_dec = None
        
        # forward thru itransformer: [B, L, C] -> [B, pred_len, C]
        output = self.model(x, x_mark_enc, x_dec, x_mark_dec)
        
        # transpose back to [B, C, pred_len] for fredf
        output = output.transpose(1, 2)
        
        return output  

class FredF(nn.Module):
    # fredf model with configurable backbone (itransformer or tsmixer)
    # lookback_window: num of past timesteps
    # forecast_horizon: how many steps to predict
    # covariates: num of parallel time series
    # backbone: 'itransformer' or 'tsmixer'
    # d_model, n_heads, e_layers, d_ff, dropout: model hyperparams
    def __init__(self, covariates: int, lookback_window: int, forecast_horizon: int,
                 backbone: str = 'itransformer',
                 d_model: int = 128, n_heads: int = 8, e_layers: int = 2, 
                 d_ff: int = 128, dropout: float = 0.1):
        super(FredF, self).__init__()
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.covariates = covariates
        self.backbone_type = backbone.lower()
        
        if self.backbone_type == 'itransformer':
            self.backbone = ITransformerWrapper(
                seq_len=lookback_window,
                pred_len=forecast_horizon,
                enc_in=covariates,
                dec_in=covariates,
                d_model=d_model,
                n_heads=n_heads,
                e_layers=e_layers,
                d_ff=d_ff,
                dropout=dropout
            )
        elif self.backbone_type == 'tsmixer':
            self.backbone = TSMixer(
                seq_len=lookback_window,
                pred_len=forecast_horizon,
                enc_in=covariates,
                dec_in=covariates,
                n_blocks=e_layers,  # Use e_layers for number of mixer blocks
                d_ff=d_ff,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown backbone '{backbone}'. Choose 'itransformer' or 'tsmixer'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.covariates and x.shape[2] == self.lookback_window, \
            f"Input shape must be (batch_size (any), {self.covariates} (same as covariates), {self.lookback_window} (same as lookback_window))" \
            f"but got {x.shape}"
        logits = self.backbone(x)
        return logits


MODEL_REGISTRY = {
    "fredf": FredF,
    "tsmixer": TSMixer,
}


def _normalize_kwargs(model_key: str, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(model_kwargs)
    if model_key == "fredf":
        alias_map = {
            "seq_len": "lookback_window",
            "pred_len": "forecast_horizon",
            "enc_in": "covariates",
            "dec_in": "covariates",
        }
        for alias, target in alias_map.items():
            if alias in normalized and target not in normalized:
                normalized[target] = normalized[alias]
    return normalized


def build_model(model_name: str, **model_kwargs: Any) -> nn.Module:

    if not model_name:
        raise ValueError("model_name must be a non-empty string")

    model_key = model_name.lower()
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}")

    model_cls = MODEL_REGISTRY[model_key]
    normalized_kwargs = _normalize_kwargs(model_key, model_kwargs)

    signature = inspect.signature(model_cls)
    supports_var_kw = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
    )

    if supports_var_kw:
        ctor_kwargs = normalized_kwargs
    else:
        ctor_kwargs = {
            name: normalized_kwargs[name]
            for name in signature.parameters.keys()
            if name in normalized_kwargs
        }

    return model_cls(**ctor_kwargs)

