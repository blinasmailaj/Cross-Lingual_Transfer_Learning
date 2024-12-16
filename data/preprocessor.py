from typing import List, Dict, Any  # Add this import at the top
import nltk
import re
from bs4 import BeautifulSoup
import unicodedata


class TextPreprocessor:
    def __init__(self):
        try:
            nltk.download('punkt')
            nltk.download('stopwords')
            self.stopwords = set(nltk.corpus.stopwords.words('english'))
        except:
            self.stopwords = set()

    def clean_text(self, text: str) -> str:
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()

    def preprocess_for_training(
        self, 
        texts: List[str], 
        summaries: List[str], 
        max_text_length: int, 
        max_summary_length: int
    ) -> Dict[str, List[str]]:
        processed_texts = []
        processed_summaries = []
        
        for text, summary in zip(texts, summaries):
            # Clean and truncate text
            cleaned_text = self.clean_text(text)
            sentences = self.segment_sentences(cleaned_text)
            
            # Keep most important sentences (based on position and length)
            important_sentences = self.select_important_sentences(sentences)
            truncated_text = ' '.join(important_sentences)[:max_text_length]
            
            # Clean and truncate summary
            cleaned_summary = self.clean_text(summary)
            truncated_summary = self.truncate_text(cleaned_summary, max_summary_length)
            
            processed_texts.append(truncated_text)
            processed_summaries.append(truncated_summary)
            
        return {
            "texts": processed_texts,
            "summaries": processed_summaries
        }

    def select_important_sentences(self, sentences: List[str]) -> List[str]:
        # Simple importance scoring based on sentence length and position
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            length_score = min(len(sentence.split()) / 20.0, 1.0)
            position_score = 1.0 - (i / len(sentences))
            score = (length_score + position_score) / 2
            scored_sentences.append((score, sentence))
        
        # Sort by score and take top sentences
        scored_sentences.sort(reverse=True)
        return [sent for _, sent in scored_sentences[:5]]