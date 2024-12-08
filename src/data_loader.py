import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import logging

class MultilingualSummarizationDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_input_length=512, max_output_length=150):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        article = self.dataset[idx]['article']
        summary = self.dataset[idx]['highlights']

        inputs = self.tokenizer(
            article, 
            max_length=self.max_input_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )

        outputs = self.tokenizer(
            summary, 
            max_length=self.max_output_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': outputs['input_ids'].flatten()
        }

def prepare_multilingual_data(config):
    """
    Load and prepare dataset for multilingual summarization with error handling
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load English dataset with specific configuration
        logger.info("Loading CNN/DailyMail dataset...")
        dataset = load_dataset(
            'cnn_dailymail',
            '3.0.0',
            cache_dir=str(config.data_dir)  # Specify cache directory
        )
        logger.info("Dataset loaded successfully")
        
        # Load multilingual tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            'google/mt5-base',
            cache_dir=str(config.data_dir)  # Specify cache directory
        )
        logger.info("Tokenizer loaded successfully")

        # Create custom datasets
        logger.info("Creating dataset splits...")
        train_data = MultilingualSummarizationDataset(
            dataset['train'], 
            tokenizer,
            max_input_length=config.max_input_length,
            max_output_length=config.max_output_length
        )
        val_data = MultilingualSummarizationDataset(
            dataset['validation'], 
            tokenizer,
            max_input_length=config.max_input_length,
            max_output_length=config.max_output_length
        )
        test_data = MultilingualSummarizationDataset(
            dataset['test'], 
            tokenizer,
            max_input_length=config.max_input_length,
            max_output_length=config.max_output_length
        )

        # Create DataLoaders
        logger.info("Creating dataloaders...")
        train_loader = DataLoader(
            train_data, 
            batch_size=config.batch_size, 
            shuffle=True,
            num_workers=0  # Set to 0 to avoid potential multiprocessing issues
        )
        val_loader = DataLoader(
            val_data, 
            batch_size=config.batch_size, 
            shuffle=False,
            num_workers=0
        )
        test_loader = DataLoader(
            test_data, 
            batch_size=config.batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        logger.info("Data preparation completed successfully")
        return train_loader, val_loader, test_loader, tokenizer

    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        raise
