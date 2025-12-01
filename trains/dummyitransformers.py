import torch
import torch.nn as nn
import torch.optim as optim
from types import SimpleNamespace
import os
import time
from torch.utils.data import Dataset, DataLoader # Added Dataset and DataLoader imports

from models.iTransformer.model import Model
# from utils.dataloader import get_train_dataloader, get_val_dataloader # Removed unused imports
from utils.losses import fredf_loss
from utils.fftlayer import FFTLayer
from colorama import Fore, Style

# dummy dataset for testing
class DummyDataset(Dataset):
    # dummy dataset that generates random data in the format the model expects
    # shape: (batch, seq_len, num_features)
    def __init__(self, seq_len, pred_len, num_samples, num_features):
        self.num_samples = num_samples
        # generate all data at once and store in memory
        self.data_x = torch.randn(num_samples, seq_len, num_features)
        self.data_y = torch.randn(num_samples, pred_len, num_features)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]


# inner for loop for each epoch
def train_one_epoch(model, train_loader, optimizer, criterion, fft_layer, device):

    model.train() # set the nn.Module model into training mode 
    total_loss = 0.0
    
    for i, (batch_x, batch_y) in enumerate(train_loader):

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device) # batch_y is time domain

        # get model prediction. note: model() returns 2 values when output_attention=True, and 1 when output_attention=False
        # handle model output: expected a tuple (y_hat, attns) but output_attention=False means only y_hat
        output = model(batch_x, None, None, None)
        y_hat_freq = output[0] if isinstance(output, tuple) else output # fix for previous ValueError

        y_target = batch_y # changed from y_acc for clarity
        if fft_layer:
            y_target = fft_layer(y_target) # use y_target

        loss = criterion(y_hat_freq, y_target) # use y_target

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(train_loader)

# validation function for a batch
def validate(model, val_loader, criterion, fft_layer, device):
    
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(val_loader):

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # handle model output
            output = model(batch_x, None, None, None)
            y_hat_freq = output[0] if isinstance(output, tuple) else output
            
            y_target = batch_y # changed from y_acc

            if fft_layer: # if there is a fft layer passed in
                y_target = fft_layer(y_target) # use y_target
            
            loss = criterion(y_hat_freq, y_target) # use y_target
            total_loss += loss.item()

    return total_loss / len(val_loader)


# training code for itransformer with fredf loss
# the fredf_loss function automatically handles fft transformation internally,
# so the model always outputs time-domain predictions
def train():

    configs = SimpleNamespace(
        # model config
        seq_len=96, # sequence length
        pred_len=24, # prediction length
        d_model=512, # model dimensionality
        n_heads=8, # num of attention heads
        e_layers=2, # num of encoder layers
        d_ff=2048, # ffn dimension
        dropout=0.1, # dropout rate
        activation='gelu', # activation function
        use_norm=True,  # normalizes input for each time series in the batch, and normalizes output (scales output back to original range)
        # note: jia's dataloader has the normalize field which applies
        # global normalization, while use_norm applies nomralization to a single batch instance for the model inside forecast(). should be fine to use both

        output_attention=False, # whether the model returns self attn weights with preds. returns (predictions, attn_weights) if True and predictions if False
                                # set to False for dummy training to save memory and avoid unpacking errors

        # note: embed and freq do not change any internal logic, use these
        # these fields are saved for additional changes later
        embed='fixed',
        freq='h',
        # added fields to prevent crashing for model initialization and device print
        factor=1, # required by the AttentionLayer in the model definition, a dummy value here

        # dummy data config
        num_features=7, # example num of time series
        num_train_samples=400,
        num_val_samples=80,

        # training configs
        batch_size=32,
        learning_rate=1e-4,
        epochs=5, # reduced for a quick test run

        use_gpu=True,
        fourier_weight=0.5
    )

    if configs.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"{Fore.GREEN}Using GPU: {torch.cuda.get_device_name(0)}{Style.RESET_ALL}")
    else:
        device = torch.device("cpu")
        print(f"{Fore.YELLOW}Using CPU{Style.RESET_ALL}")

    # model init
    print("Initializing model")
    model = Model(configs).to(device)
    
    # dummy dataloader init
    print("Loading dummy data...")
    train_dataset = DummyDataset(configs.seq_len, configs.pred_len, configs.num_train_samples, configs.num_features)
    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True)
    
    val_dataset = DummyDataset(configs.seq_len, configs.pred_len, configs.num_val_samples, configs.num_features)
    val_loader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate)
    
    fft_layer = None
    # use a standard loss by default
    criterion = nn.MSELoss()
    # conditionally set up for frequency domain training
    if configs.output_transformation == 'freq':
        # use a lambda to wrap the criterion with its fourier_weight argument
        criterion = lambda pred, target: fourier_mse_loss(pred, target, fourier_weight=configs.fourier_weight)
        fft_layer = FFTLayer().to(device)
    
    print(f"{Fore.CYAN}--- Starting Dummy Training ---{Style.RESET_ALL}")
    print(f"Model: iTransformer | Epochs: {configs.epochs} | Loss: FreDF (λ={configs.fourier_weight})")
    
    # check if criterion is a lambda before trying to access __name__
    if hasattr(criterion, '__name__'):
        print(f"Loss function: '{criterion.__name__}'")
    else:
        print(f"Loss function: 'Custom Lambda Loss'")
    
    print(f"-------------------------")

    for epoch in range(configs.epochs): # training loop
        epoch_start_time = time.time()
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, fft_layer, device)
        val_loss = validate(model, val_loader, criterion, fft_layer, device)
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        print(
            f"Epoch: {epoch + 1:02} | "
            f"Time: {epoch_duration:.2f}s | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

    print(f"{Fore.GREEN}--- Dummy Training Complete ---{Style.RESET_ALL}")

if __name__ == "__main__":
    train()
