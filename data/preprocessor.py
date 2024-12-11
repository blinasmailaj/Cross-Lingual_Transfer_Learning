import re
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Union
import html
from bs4 import BeautifulSoup
import unicodedata

class TextPreprocessor:
    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def clean_text(self, text: str) -> str:
        """Clean raw text by removing unwanted elements and normalizing."""
        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()
        
        # Unescape HTML entities
        text = html.unescape(text)
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text.strip()

    def segment_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        return sent_tokenize(text)

    def truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to max_length while keeping complete sentences."""
        sentences = self.segment_sentences(text)
        truncated_text = ""
        
        for sentence in sentences:
            if len(truncated_text) + len(sentence) + 1 <= max_length:
                truncated_text += sentence + " "
            else:
                break
                
        return truncated_text.strip()

    def preprocess_for_training(
        self, 
        texts: List[str], 
        summaries: List[str], 
        max_text_length: int, 
        max_summary_length: int
    ) -> Dict[str, List[str]]:
        """Preprocess texts and summaries for training."""
        processed_texts = []
        processed_summaries = []
        
        for text, summary in zip(texts, summaries):
            # Clean and truncate text
            cleaned_text = self.clean_text(text)
            truncated_text = self.truncate_text(cleaned_text, max_text_length)
            
            # Clean and truncate summary
            cleaned_summary = self.clean_text(summary)
            truncated_summary = self.truncate_text(cleaned_summary, max_summary_length)
            
            processed_texts.append(truncated_text)
            processed_summaries.append(truncated_summary)
            
        return {
            "texts": processed_texts,
            "summaries": processed_summaries
        }
