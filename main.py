import torch
import numpy as np
import random
from transformers import MT5Tokenizer, AdamW
from config import get_config
from data.data_loader import get_datasets, create_data_loaders
from models.summarizer import CrossLingualSummarizer
from train import train_model
import wandb
from predict import batch_predict
import os
from bs4 import BeautifulSoup

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    config = get_config()
    
    set_seed(config['seed'])
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['output_dir'], exist_ok=True)
    
    device = torch.device(config['device'])

    
    tokenizer = MT5Tokenizer.from_pretrained(config['model_name'])
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Load datasets
    print("Loading datasets...")
    train_data, val_data, test_data = get_datasets(config)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data, tokenizer, config
    )
    
    # Initialize model
    print("Initializing model...")
    model = CrossLingualSummarizer(config).to(device)
    model.model.resize_token_embeddings(len(tokenizer))
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    
    # Train model
    print("Starting training...")
    train_model(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        tokenizer=tokenizer,
    )
    config = get_config()
    
    set_seed(config['seed'])
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['output_dir'], exist_ok=True)
    
    if config['use_wandb']:
        wandb.init(project=config['wandb_project'], config=config)
    
    device = torch.device(config['device'])
    
    tokenizer = MT5Tokenizer.from_pretrained(config['model_name'])
    
    print("Loading datasets...")
    train_data, val_data, test_data = get_datasets(config)
    
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data, tokenizer, config
    )
    
    print("Initializing model...")
    model = CrossLingualSummarizer(config).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Train model
    print("Starting training...")
    train_model(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        tokenizer=tokenizer,
    )
    
    print("Performing final evaluation...")
    checkpoint = torch.load(os.path.join(config['checkpoint_dir'], 'final_model_20241218_054329.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Generating example predictions...")
    example_texts = [item['article'] for item in test_data[:5]]
    predictions = batch_predict(model, tokenizer, example_texts, config)
    
    for i, (text, pred) in enumerate(zip(example_texts, predictions)):
        print(f"\nExample {i+1}:")
        print(f"Original: {text[:200]}...")
        print(f"Summary: {pred}")
    
    if config['use_wandb']:
        wandb.finish()
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
