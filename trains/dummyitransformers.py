import torch
import torch.nn as nn
import torch.optim as optim
from types import SimpleNamespace
import os
import time
from torch.utils.data import Dataset, DataLoader

# --- Model and Utilities ---
# Assuming this script is run from the project root (CS4451)
from models.iTransformer.model import Model
from utils.losses import fourier_mse_loss
from utils.fftlayer import FFTLayer

# --- For nice printing ---
from colorama import Fore, Style

# --- Dummy Dataset ---
class DummyDataset(Dataset):
    """
    A dummy dataset that generates random data in the required shape.
    """
    def __init__(self, seq_len, pred_len, num_samples, num_features):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_samples = num_samples
        self.num_features = num_features
        
        # Generate all data at once and store in memory
        # Shape for iTransformer input (batch, features, seq_len)
        self.data_x = torch.randn(num_samples, num_features, seq_len)
        self.data_y = torch.randn(num_samples, num_features, pred_len)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]

# --- Training and Validation Functions (copied from your script) ---
def train_one_epoch(model, train_loader, optimizer, criterion, fft_layer, device):
    """
    Performs one full training epoch.
    """
    model.train()
    total_loss = 0.0
    
    for i, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device) # batch_y is time domain 

        y_hat_freq, _ = model(batch_x, None, None, None)

        y_acc = batch_y
        if fft_layer:
            y_acc = fft_layer(batch_y)

        loss = criterion(y_hat_freq, y_acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, fft_layer, device):
    """
    Performs validation on the validation set.
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(val_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_hat_freq, _ = model(batch_x, None, None, None)
            
            y_acc = batch_y 
            if fft_layer:
                y_acc = fft_layer(batch_y)
            
            loss = criterion(y_hat_freq, y_acc)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def train():
    """
    Main training function using dummy data.
    """
    configs = SimpleNamespace(
        # Model config
        seq_len=96,
        pred_len=24,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_ff=2048,
        dropout=0.1,
        activation='gelu',
        output_transformation='freq',
        use_norm=True,
        output_attention=False, # Set to False for dummy training to save memory
        embed='fixed',
        freq='h',
        factor=5, # Added missing factor for AttentionLayer

        # Dummy Data configs
        num_features=7, # Number of variates in the dummy data
        num_train_samples=500,
        num_val_samples=100,

        # Training configs
        batch_size=32,
        learning_rate=1e-4,
        epochs=5, # Reduced for a quick test run

        # System configs
        use_gpu=True,
        gpu_id=0
    )

    # --- Setup Device ---
    if configs.use_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{configs.gpu_id}")
        print(f"{Fore.GREEN}Using GPU: {torch.cuda.get_device_name(configs.gpu_id)}{Style.RESET_ALL}")
    else:
        device = torch.device("cpu")
        print(f"{Fore.YELLOW}Using CPU{Style.RESET_ALL}")

    # --- Initialize Model ---
    print("Initializing model...")
    model = Model(configs).to(device)
    
    # --- Create Dummy Dataloaders ---
    print("Creating dummy data...")
    train_dataset = DummyDataset(configs.seq_len, configs.pred_len, configs.num_train_samples, configs.num_features)
    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True)
    
    val_dataset = DummyDataset(configs.seq_len, configs.pred_len, configs.num_val_samples, configs.num_features)
    val_loader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False)

    # --- Optimizer, Criterion, etc. ---
    optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate)
    
    fft_layer = None
    criterion = nn.MSELoss()
    if configs.output_transformation == 'freq':
        print(f"{Fore.CYAN}Training in frequency domain. Using fourier_mse_loss.{Style.RESET_ALL}")
        criterion = fourier_mse_loss
        fft_layer = FFTLayer().to(device)
    else:
        print(f"{Fore.CYAN}Training in time domain. Using nn.MSELoss.{Style.RESET_ALL}")

    # --- Start Training ---
    print(f"--- Starting Dummy Training ---")
    for epoch in range(configs.epochs):
        epoch_start_time = time.time()
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, fft_layer, device)
        val_loss = validate(model, val_loader, criterion, fft_layer, device)
        
        epoch_duration = time.time() - epoch_start_time
        
        print(
            f"Epoch: {epoch + 1:02} | "
            f"Time: {epoch_duration:.2f}s | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

    print(f"{Fore.GREEN}--- Dummy Training Complete ---{Style.RESET_ALL}")

if __name__ == "__main__":
    train()
