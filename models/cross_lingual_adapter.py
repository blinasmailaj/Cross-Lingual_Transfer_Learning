import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossLingualAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Increased adapter size for better cross-lingual transfer
        self.down_project = nn.Linear(config.d_model, config.d_model // 2)
        self.up_project = nn.Linear(config.d_model // 2, config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        # Enhanced language-specific parameters
        self.language_embeddings = nn.Parameter(torch.randn(2, config.d_model))
        self.language_specific_layer_norm = nn.LayerNorm(config.d_model)
        
        # Additional cross-lingual alignment layer
        self.alignment_layer = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, hidden_states, language_id=0):
        residual = hidden_states
        
        # Down projection
        hidden_states = self.down_project(hidden_states)
        
        # Add language-specific information
        language_emb = self.language_embeddings[language_id]
        hidden_states = hidden_states + language_emb.unsqueeze(0).unsqueeze(0)
        
        # Cross-lingual alignment
        hidden_states = self.alignment_layer(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Up projection
        hidden_states = self.up_project(hidden_states)
        
        # Residual connection and layer normalization
        hidden_states = self.layer_norm(hidden_states + residual)
        hidden_states = self.language_specific_layer_norm(hidden_states)
        
        return hidden_states