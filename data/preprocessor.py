import nltk
import re
import unicodedata
from bs4 import BeautifulSoup
from typing import List, Dict, Any

class TextPreprocessor:
    def __init__(self):
        try:
            nltk.download('punkt')
            nltk.download('stopwords')
            self.stopwords = {
                'en': set(nltk.corpus.stopwords.words('english')),
                'sq': set()  # Add Albanian stopwords if available
            }
        except:
            self.stopwords = {'en': set(), 'sq': set()}

    def clean_text(self, text: str, language: str) -> str:
        # Language-specific preprocessing
        if language == 'sq':  # Albanian
            # Handle Albanian-specific characters
            text = unicodedata.normalize('NFKC', text)
            # Keep Albanian-specific punctuation and characters
            text = re.sub(r'[^\w\s.,!?;:"-ëçÇËÀ]', '', text)
        else:
            # English preprocessing
            text = unicodedata.normalize('NFKD', text)
            text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Common preprocessing steps
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = BeautifulSoup(text, "html.parser").get_text()
        
        # Language-specific case handling
        if language == 'sq':
            # Preserve case for Albanian
            text = ' '.join(text.split())
        else:
            # Convert to lowercase for English
            text = text.lower()
            text = ' '.join(text.split())
        
        return text.strip()

    def segment_sentences(self, text: str, language: str) -> List[str]:
        """Language-aware sentence segmentation"""
        try:
            if language == 'sq':
                # Add Albanian-specific sentence boundary rules
                sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
                sent_detector._params.sent_starters.update({
                    'gjithashtu': True,
                    'megjithatë': True,
                    'sidoqoftë': True
                })
                return sent_detector.tokenize(text)
            else:
                return nltk.sent_tokenize(text)
        except:
            return [text]
