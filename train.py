# MODULE: Swarms
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset  # Hugging Face datasets
from loguru import logger  # Loguru for logging
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm  # Progress bar for training

from omni_byte_former.model import ByteTransformer

# Set up logging with Loguru
logger.add("training.log", rotation="500 MB")  # Logs saved to file with rotation

# Define EOS token (value 255 for end-of-sequence)
EOS_TOKEN = 255

#################################
# Model Definition
#################################


#################################
# Hugging Face Dataset Wrapper
#################################

class HuggingFaceTextDataset(Dataset):
    def __init__(self, data, seq_len=100):
        """
        A dataset wrapper to convert Hugging Face text data into byte sequences.

        Args:
            data (datasets.Dataset): Hugging Face dataset.
            seq_len (int): Maximum sequence length (input and target will be truncated/padded to this length).
        """
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the text from Hugging Face dataset
        text = self.data[idx]['text']
        byte_data = byteify(text)

        # Truncate or pad byte_data to seq_len
        byte_tensor = bytes_to_tensor(byte_data, self.seq_len)
        
        # Input (src) is the original text, Target (tgt) is the shifted sequence
        src = byte_tensor[:-1]
        tgt = byte_tensor[1:]

        return src, tgt

def byteify(data: str) -> bytes:
    return data.encode('utf-8')

def bytes_to_tensor(byte_data: bytes, seq_len: int) -> torch.Tensor:
    byte_list = list(byte_data)[:seq_len]
    if len(byte_list) < seq_len:
        byte_list += [EOS_TOKEN] * (seq_len - len(byte_list))  # Padding with EOS
    return torch.tensor(byte_list, dtype=torch.long)

#################################
# Training Function
#################################

def train(model, dataloader, optimizer, criterion, device, epochs, save_path):
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        # Use tqdm to create a progress bar
        for i, (src, tgt) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()
            output = model(src, tgt)
            
            # Loss function only needs to compare with target tokens, so we ignore padding tokens
            loss = criterion(output.view(-1, 256), tgt.view(-1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Log epoch loss
        avg_loss = running_loss / len(dataloader)
        logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # Save checkpoint every epoch
        save_checkpoint(model, optimizer, epoch, save_path)

#################################
# Evaluation Function
#################################

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt)

            loss = criterion(output.view(-1, 256), tgt.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    logger.info(f"Validation Loss: {avg_loss:.4f}")

#################################
# Utility: Save and Load Checkpoints
#################################

def save_checkpoint(model, optimizer, epoch, path):
    """
    Save model and optimizer state as a checkpoint.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, os.path.join(path, f"model_epoch_{epoch}.pth"))
    logger.info(f"Model checkpoint saved at epoch {epoch}.")

def load_checkpoint(model, optimizer, path):
    """
    Load model and optimizer state from a checkpoint.
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    logger.info(f"Loaded model from {path}, resuming from epoch {start_epoch}.")
    return model, optimizer, start_epoch

#################################
# Main Training Loop
#################################

def main(args):
    # Load Hugging Face wikitext-2 dataset
    logger.info("Loading dataset from Hugging Face...")
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
    val_dataset = load_dataset("wikitext", "wikitext-2-v1", split="validation")

    # Wrap dataset in a PyTorch Dataset
    train_dataset = HuggingFaceTextDataset(dataset, seq_len=args.seq_len)
    val_dataset = HuggingFaceTextDataset(val_dataset, seq_len=args.seq_len)

    # Prepare DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ByteTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=EOS_TOKEN)

    # Start training
    logger.info(f"Starting training for {args.epochs} epochs...")
    train(model, train_loader, optimizer, criterion, device, args.epochs, args.save_path)

    # Evaluate the model
    logger.info("Evaluating the model...")
    evaluate(model, val_loader, criterion, device)

#################################
# Argument Parsing for CLI
#################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ByteTransformer on Hugging Face dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seq_len", type=int, default=128, help="Max sequence length")
    parser.add_argument("--save_path", type=str, default="./checkpoints", help="Path to save model checkpoints")
    args = parser.parse_args()

    # Ensure save path exists
    os.makedirs(args.save_path, exist_ok=True)

    # Run the main training loop
    main(args)
