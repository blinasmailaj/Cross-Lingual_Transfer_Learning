import torch

def get_config():
    return {
        # Model Configuration
        'model_name': 'google/mt5-base',
        'max_length': 128,
        'batch_size': 8,
        'gradient_accumulation_steps': 2,
        
        # Training Configuration
        'learning_rate': 3e-5,
        'num_epochs': 3,
        'warmup_ratio': 0.1,
        'max_grad_norm': 1.0,
        'weight_decay': 0.01,
        'early_stopping_patience': 3,
        
        # Dataset Configuration
        'train_size': 7000,
        'val_size': 1000,
        'test_size': 500,
        
        # System Configuration
        'num_workers': 0,
        'distributed': False,
        'fp16': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42,
        
        # Directory Configuration
        'checkpoint_dir': 'checkpoints',  # Added this line
        'output_dir': 'outputs',
        'log_dir': 'logs',  # Optional, added for completeness
        
        # Additional Configuration
        'use_wandb': False,
        'wandb_project': None
    }