from rouge_score import rouge_scorer
import numpy as np
from typing import Dict, List
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import nltk
try:
    nltk.download('punkt')
except:
    pass

def compute_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores with better preprocessing"""
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True,
        tokenizer=nltk.word_tokenize
    )
    
    scores = {
        'rouge1': [],
        'rouge2': [],
        'rougeL': []
    }
    
    for pred, ref in zip(predictions, references):
        # Clean the texts
        pred = pred.strip().lower()
        ref = ref.strip().lower()
        
        # Compute scores
        score = scorer.score(ref, pred)
        scores['rouge1'].append(score['rouge1'].fmeasure)
        scores['rouge2'].append(score['rouge2'].fmeasure)
        scores['rougeL'].append(score['rougeL'].fmeasure)
    
    # Calculate means
    return {
        'rouge1': np.mean(scores['rouge1']),
        'rouge2': np.mean(scores['rouge2']),
        'rougeL': np.mean(scores['rougeL']),
        'rouge_avg': np.mean([
            np.mean(scores['rouge1']),
            np.mean(scores['rouge2']),
            np.mean(scores['rougeL'])
        ])
    }

def compute_bleu_score(predictions: List[str], references: List[str]) -> float:
    """Compute BLEU score for additional evaluation"""
    # Tokenize predictions and references
    pred_tokens = [nltk.word_tokenize(pred.lower()) for pred in predictions]
    ref_tokens = [[nltk.word_tokenize(ref.lower())] for ref in references]
    
    # Compute BLEU score with smoothing
    smoothing = SmoothingFunction()
    return corpus_bleu(
        ref_tokens,
        pred_tokens,
        smoothing_function=smoothing.method1
    )

def compute_language_metrics(predictions: List[str], references: List[str], language: str) -> Dict[str, float]:
    """Compute metrics"""
    # Get ROUGE scores
    rouge_scores = compute_rouge_scores(predictions, references)
    
    # Get BLEU score
    bleu_score = compute_bleu_score(predictions, references)
    
    # Combine metrics
    metrics = {
        **rouge_scores,
        'bleu': bleu_score,
        'language': language
    }
    
    return metrics