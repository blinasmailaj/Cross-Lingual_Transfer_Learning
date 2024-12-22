import torch
import torch.nn as nn
from transformers import MT5ForConditionalGeneration
from typing import Dict, Any, Optional, Tuple, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrossLingualSummarizer(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Load base MT5 model
        logger.info(f"Loading MT5 model from {config['model_name']}...")
        self.model = MT5ForConditionalGeneration.from_pretrained(config['model_name'])
        
        # Initialize language-specific adapters
        d_model = self.model.config.d_model
        self.en_adapter = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model)
        )
        
        self.sq_adapter = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Language embeddings
        self.language_embeddings = nn.Embedding(2, d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        language_id: Optional[torch.Tensor] = None
    ) -> Union[Tuple[torch.Tensor, ...], Any]:
        # Forward pass through MT5
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs

    def generate_summary(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        language_id: Optional[torch.Tensor] = None,
        num_beams: int = 4,
        max_length: int = 150,
        min_length: int = 30,
        length_penalty: float = 2.0,
        early_stopping: bool = True,
        no_repeat_ngram_size: int = 3
    ) -> torch.Tensor:
        """Generate summaries"""
        try:
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                no_repeat_ngram_size=no_repeat_ngram_size
            )
        except Exception as e:
            logger.error(f"Error in summary generation: {str(e)}")
            raise

    def save_model(self, path: str) -> None:
        """Save the model"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'en_adapter_state_dict': self.en_adapter.state_dict(),
                'sq_adapter_state_dict': self.sq_adapter.state_dict(),
                'language_embeddings_state_dict': self.language_embeddings.state_dict(),
                'config': self.config
            }, path)
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, path: str) -> None:
        """Load the model"""
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.en_adapter.load_state_dict(checkpoint['en_adapter_state_dict'])
            self.sq_adapter.load_state_dict(checkpoint['sq_adapter_state_dict'])
            self.language_embeddings.load_state_dict(checkpoint['language_embeddings_state_dict'])
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
