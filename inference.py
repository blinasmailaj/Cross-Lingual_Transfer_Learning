import torch
from transformers import MT5Tokenizer
from models.summarizer import CrossLingualSummarizer
from config import get_config
import os
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SummarizerInference:
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        """
        Initialize the inference model
        Args:
            checkpoint_path: Path to the saved model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.config = get_config()
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        logger.info(f"Initializing inference model on {self.device}")
        
        # Load model and tokenizer
        self.model = self._load_model(checkpoint_path)
        self.tokenizer = MT5Tokenizer.from_pretrained(self.config['model_name'])
        
        logger.info("Model and tokenizer loaded successfully")

    def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
        """Load the model from checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        model = CrossLingualSummarizer(self.config).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model

    def generate_summary(self, text: str) -> str:
        """
        Generate summary for input text
        Args:
            text: Input text to summarize
        Returns:
            Generated summary
        """
        # Tokenize input text
        inputs = self.tokenizer(
            text,
            max_length=self.config['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            generated_ids = self.model.generate_summary(
                inputs['input_ids'],
                inputs['attention_mask']
            )
            
        summary = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return summary

    def batch_generate_summaries(self, texts: list[str]) -> list[str]:
        """
        Generate summaries for multiple texts
        Args:
            texts: List of input texts
        Returns:
            List of generated summaries
        """
        summaries = []
        for text in texts:
            summary = self.generate_summary(text)
            summaries.append(summary)
        return summaries

# Example usage
if __name__ == "__main__":
    # Path to your saved model checkpoint
    checkpoint_path = "checkpoints/best_model.pt"
    
    # Initialize inference
    summarizer = SummarizerInference(checkpoint_path)
    
    # Example texts
    texts = [
        "Your first article text here.",
        "Your second article text here."
    ]
    
    # Generate summaries
    summaries = summarizer.batch_generate_summaries(texts)
    
    # Print results
    for i, (text, summary) in enumerate(zip(texts, summaries), 1):
        print(f"\nArticle {i}:")
        print(f"Original: {text[:200]}...")
        print(f"Summary: {summary}")