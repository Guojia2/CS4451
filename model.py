import torch
from torch import nn
from torch.utils.data import DataLoader

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
