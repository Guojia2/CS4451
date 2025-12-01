import torch
import torch.nn as nn
import torch.optim as optim
from types import SimpleNamespace
import os
import time

from models.iTransformer.model import Model
from utils.dataloader import get_train_dataloader, get_val_dataloader
from utils.losses import fredf_loss
from utils.fftlayer import FFTLayer
from colorama import Fore, Style

# inner for loop for each epoch
def train_one_epoch(model, train_loader, optimizer, criterion, device):

    model.train() # set the nn.Module model into training mode
    total_loss = 0.0
    
    for i, (batch_x, batch_y) in enumerate(train_loader):

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device) # batch_y is time domain

        # get model prediction
        output = model(batch_x, None, None, None)
        y_hat = output[0] if isinstance(output, tuple) else output

        # fredf_loss handles both time and frequency domain internally
        loss = criterion(y_hat, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(train_loader)

# validation function for a batch
def validate(model, val_loader, criterion, device):
    
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(val_loader):

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            output = model(batch_x, None, None, None)
            y_hat = output[0] if isinstance(output, tuple) else output 
            
            # fredf_loss handles both time and frequency domain internally
            loss = criterion(y_hat, batch_y)
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

        output_attention=True, # whether the model returns self attn weights with preds. returns (predictions, attn_weights) if True and predictions if False

        # note: embed and freq do not change any internal logic, use these
        # these fields are saved for additional changes later
        embed='fixed',
        freq='h',

        # note: replace this with actual dataset and batch size
        dataset_name='ETTh1',
        root_path='./data/',
        batch_size=32,

        learning_rate=1e-4,
        epochs=10,

        use_gpu=True,
        fourier_weight=0.5
    )

    if configs.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"{Fore.GREEN}Using GPU: {torch.cuda.get_device_name(0)}{Style.RESET_ALL}")
    else:
        device = torch.device("cpu")
        print(f"{Fore.YELLOW}Using CPU{Style.RESET_ALL}")

    if not os.path.exists(configs.root_path):
        print(f"{Fore.RED}Data directory not found at '{configs.root_path}'. Please create it and add your datasets.{Style.RESET_ALL}")
        return

    print("Initializing model")
    model = Model(configs).to(device)
    
    # load data
    print("Loading data...")
    train_loader = get_train_dataloader(configs.seq_len, configs.pred_len, configs.batch_size, dataset_name=configs.dataset_name, root_path=configs.root_path)
    val_loader = get_val_dataloader(configs.seq_len, configs.pred_len, configs.batch_size, dataset_name=configs.dataset_name, root_path=configs.root_path)

    optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate)
    
    # use fredf_loss which handles fft internally
    criterion = lambda pred, target: fredf_loss(pred, target, fourier_weight=configs.fourier_weight)
    

    print(f"{Fore.CYAN}--- Starting Training ---{Style.RESET_ALL}")
    print(f"Model: iTransformer | Epochs: {configs.epochs} | Dataset: {configs.dataset_name}")
    print(f"Loss function: FreDF (λ={configs.fourier_weight})")
    print(f"-------------------------")

    for epoch in range(configs.epochs): # training loop; todo: add early stopping check using validation loss here
        epoch_start_time = time.time()
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        print(
            f"Epoch: {epoch + 1:02} | "
            f"Time: {epoch_duration:.2f}s | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

    print(f"{Fore.GREEN}--- Training Complete ---{Style.RESET_ALL}")

    # todo: save the model

if __name__ == "__main__":
    train()
