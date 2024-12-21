# models/summarizer.py
import torch
import torch.nn as nn
from transformers import MT5ForConditionalGeneration
from typing import Dict, Any, Optional, Tuple, Union
import logging
from .cross_lingual_adapter import CrossLingualAdapter  # Updated import

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
        self.adapter = CrossLingualAdapter(self.model.config)
        
        # Language-specific output layers
        self.language_output_projections = nn.ModuleDict({
            'en': nn.Linear(d_model, d_model),
            'sq': nn.Linear(d_model, d_model)
        })
        
        # Loss weights for different languages
        self.language_loss_weights = {
            'en': 1.0,
            'sq': 1.2  # Higher weight for low-resource language
        }

    def compute_language_specific_loss(self, outputs, labels, language_id):
        """Compute additional loss term for low-resource language"""
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Add language-specific penalty for rare tokens
        if language_id == 1:  # Albanian
            rare_token_mask = (labels != -100) & (labels < 1000)  # Adjust token ID range as needed
            if rare_token_mask.any():
                rare_token_loss = loss_fct(logits.view(-1, logits.size(-1))[rare_token_mask.view(-1)], 
                                         labels.view(-1)[rare_token_mask.view(-1)])
                return rare_token_loss
        return 0.0

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
            return_dict=True,
            output_hidden_states=True
        )
        
        # Apply language-specific adapter
        hidden_states = outputs.hidden_states[-1]
        adapted_hidden_states = self.adapter(hidden_states, language_id)
        
        # Apply language-specific output projection
        lang_key = 'sq' if language_id == 1 else 'en'
        final_hidden_states = self.language_output_projections[lang_key](adapted_hidden_states)
        
        # Compute loss with language-specific weighting
        if labels is not None:
            base_loss = outputs.loss
            lang_specific_loss = self.compute_language_specific_loss(outputs, labels, language_id)
            loss_weight = self.language_loss_weights[lang_key]
            total_loss = base_loss * loss_weight + 0.1 * lang_specific_loss
            outputs.loss = total_loss
        
        return outputs

def generate_summary(
    self,
    input_ids,
    attention_mask,
    language_id=None,
    **kwargs
):
    try:
        # Create proper cache for generation
        cache = None
        if hasattr(self.model, 'create_cache'):
            cache = self.model.create_cache()
        elif hasattr(self.model, 'encoder_decoder_cache'):
            cache = self.model.encoder_decoder_cache

        generation_config = self.config['generation']
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values_cache=cache if cache else None,  # Use new cache format
            **{**generation_config, **kwargs}
        )
        
        return outputs
    except Exception as e:
        logger.error(f"Error in summary generation: {str(e)}")
        raise