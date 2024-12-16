from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from typing import Dict, List, Tuple
from .preprocessor import TextPreprocessor

class SummarizationDataset(Dataset):
    """Dataset class for text summarization"""
    def __init__(self, data, tokenizer, config: Dict, is_train: bool = True):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.is_train = is_train
        self.preprocessor = TextPreprocessor()
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Clean and preprocess the text
        article = self.preprocessor.clean_text(item['article'])
        highlights = self.preprocessor.clean_text(item['highlights'])
        
        # Tokenize input text
        encoder_inputs = self.tokenizer(
            article,
            max_length=self.config['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize summary
        decoder_inputs = self.tokenizer(
            highlights,
            max_length=self.config['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        # Return tensors
        return {
            'input_ids': encoder_inputs['input_ids'].squeeze(0),
            'attention_mask': encoder_inputs['attention_mask'].squeeze(0),
            'labels': decoder_inputs['input_ids'].squeeze(0)
        }

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collation function for batching data"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def get_datasets(config: Dict) -> Tuple[Dataset, Dataset, Dataset]:
    """Load and split the dataset"""
    try:
        # Load the CNN/DailyMail dataset
        dataset = load_dataset('cnn_dailymail', '3.0.0')
        
        # Select subset of data based on config
        train_data = dataset['train'].select(range(config['train_size']))
        val_data = dataset['validation'].select(range(config['val_size']))
        test_data = dataset['test'].select(range(config['test_size']))
        
        print(f"Loaded {len(train_data)} training examples")
        print(f"Loaded {len(val_data)} validation examples")
        print(f"Loaded {len(test_data)} test examples")
        
        return train_data, val_data, test_data
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

def create_data_loaders(
    train_data: Dataset,
    val_data: Dataset,
    test_data: Dataset,
    tokenizer,
    config: Dict
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoader objects for training, validation and testing"""
    try:
        # Create datasets
        train_dataset = SummarizationDataset(train_data, tokenizer, config, is_train=True)
        val_dataset = SummarizationDataset(val_data, tokenizer, config, is_train=False)
        test_dataset = SummarizationDataset(test_data, tokenizer, config, is_train=False)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        print(f"Error creating data loaders: {str(e)}")
        raise

def get_sample_batch(data_loader: DataLoader) -> Dict[str, torch.Tensor]:
    """Get a sample batch from a data loader for testing purposes"""
    try:
        iterator = iter(data_loader)
        batch = next(iterator)
        return batch
    except Exception as e:
        print(f"Error getting sample batch: {str(e)}")
        raise