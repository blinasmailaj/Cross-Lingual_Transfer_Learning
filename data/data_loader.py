import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnglishSummarizationDataset(Dataset):
    def __init__(self, data, tokenizer, config: Dict, is_train: bool = True):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.is_train = is_train
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Add English language token
        article = f"<en> {item['article']}"
        highlights = f"<en> {item['highlights']}"
        
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
        
        return {
            'input_ids': encoder_inputs['input_ids'].squeeze(0),
            'attention_mask': encoder_inputs['attention_mask'].squeeze(0),
            'labels': decoder_inputs['input_ids'].squeeze(0),
            'language_id': torch.tensor(0)  # 0 for English
        }

def get_datasets(config: Dict) -> Tuple[Dataset, Dataset, Dataset]:
    """Load English dataset"""
    try:
        # Load the CNN/DailyMail dataset
        dataset = load_dataset('cnn_dailymail', '3.0.0')
        
        # Select subset of data
        train_data = dataset['train'].select(range(config['train_size']))
        val_data = dataset['validation'].select(range(config['val_size']))
        test_data = dataset['test'].select(range(config['test_size']))
        
        logger.info(f"Loaded {len(train_data)} training examples")
        logger.info(f"Loaded {len(val_data)} validation examples")
        logger.info(f"Loaded {len(test_data)} test examples")
        
        return train_data, val_data, test_data
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def create_data_loaders(
    train_data: Dataset,
    val_data: Dataset,
    test_data: Dataset,
    tokenizer,
    config: Dict
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training, validation and testing"""
    try:
        # Create datasets
        train_dataset = EnglishSummarizationDataset(train_data, tokenizer, config, is_train=True)
        val_dataset = EnglishSummarizationDataset(val_data, tokenizer, config, is_train=False)
        test_dataset = EnglishSummarizationDataset(test_data, tokenizer, config, is_train=False)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        logger.error(f"Error creating data loaders: {str(e)}")
        raise