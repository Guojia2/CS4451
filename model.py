import inspect
from typing import Any, Dict

import torch
from torch import nn

from models import TSMixer

class ITransformerLayer(nn.Module):
    """
    Small transformer-based module that accepts input of shape
    (batch, covariates, lookback_window) and returns
    (batch,  covariates, forecast_horizon)

    """
    def __init__(self, covariates: int, lookback_window: int, forecast_horizon: int):
        super().__init__()
        self.covariates = covariates
        self.lookback = lookback_window
        self.forecast_horizon = forecast_horizon

        # TODO TRANSFORMER IMPLEMENTER complete initialization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, covariates, lookback)
        b, c, t = x.shape
        assert (c == self.covariates and t == self.lookback)
        
        # TODO TRANSFORMER IMPLEMENTER simplement forward pass

        return None  

class FredF(nn.Module):

    """
    initialization method for FredF model
    Args:

        lookback_window (int): The number of past time steps to consider for making predictions.
        forecast_horizon (int): The number of future time step to predict
        covariates (int): Number of parallel time series in  input data.
    """
    def __init__(self, covariates: int, lookback_window: int, forecast_horizon: int):
        super(FredF, self).__init__()
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.covariates = covariates
        self.itransformer_layer = ITransformerLayer(covariates, lookback_window, forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.covariates and x.shape[2] == self.lookback_window, \
            f"Input shape must be (batch_size (any), {self.covariates} (same as covariates), {self.lookback_window} (same as lookback_window))" \
            f"but got {x.shape}"
        logits = self.itransformer_layer(x)

        
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
            """Constructs a model from the registry using normalized kwargs."""

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
