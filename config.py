import torch

def get_config():
    return {
        # Model Configuration
        'model_name': 'google/mt5-large',
        'max_length': 768,
        'batch_size': 4,
        'gradient_accumulation_steps': 8,
        'hidden_size': 512,
        
        # Training Configuration
        'learning_rate': 1e-5,
        'num_epochs': 10,
        'warmup_ratio': 0.2,
        'max_grad_norm': 0.5,
        'weight_decay': 0.01,
        'early_stopping_patience': 5,
        'label_smoothing': 0.1,
        
        # Generation Configuration
        'generation': {
            'num_beams': 5,
            'length_penalty': 1.2,
            'no_repeat_ngram_size': 2,
            'early_stopping': True,
            'do_sample': True,
            'top_k': 50,
            'top_p': 0.95,
            'temperature': 0.7,
            'diversity_penalty': 0.5,
            'num_beam_groups': 4,
            'repetition_penalty': 1.2
        },
        
        # Dataset Configuration
        'train_size': 20000,
        'val_size': 2000,
        'test_size': 1000,
        'max_source_length': 768,
        'max_target_length': 150,
        
        # Language Configuration
        'source_lang': ['en', 'sq'],
        'target_lang': ['en', 'sq'],
        'use_language_adapters': True,
        
        # System Configuration
        'num_workers': 4,
        'distributed': False,
        'fp16': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42,
        
        # Directories
        'checkpoint_dir': 'checkpoints',
        'output_dir': 'outputs',
        'log_dir': 'logs'
    }