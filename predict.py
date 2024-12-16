import torch
import json
from typing import List, Dict, Any

def load_model(checkpoint_path: str, config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    from models.summarizer import CrossLingualSummarizer
    
    model = CrossLingualSummarizer(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def generate_summary(
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    config: Dict[str, Any]
) -> str:
    model.eval()
    
    with torch.no_grad():
        inputs = tokenizer(
            text,
            max_length=config['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(config['device'])
        
        generated_ids = model.generate_summary(
            inputs['input_ids'],
            inputs['attention_mask']
        )
        
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def batch_predict(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: List[str],
    config: Dict[str, Any],
    output_file: str = None
) -> List[str]:
    summaries = []
    
    for text in texts:
        summary = generate_summary(model, tokenizer, text, config)
        summaries.append(summary)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
    
    return summaries
