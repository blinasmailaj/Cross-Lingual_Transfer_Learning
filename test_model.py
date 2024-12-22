import torch
from transformers import MT5Tokenizer
from models.summarizer import CrossLingualSummarizer
from config import get_config
import logging
import os
import re
from rouge_score import rouge_scorer
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_generated_summary(generated_text: str) -> str:
    cleaned_text = re.sub(r"<extra_id_\d+>", "", generated_text)
    cleaned_text = re.sub(r"(<\w+>)+", "", cleaned_text)
    cleaned_text = re.sub(r"(\.|!|\?|\s){2,}", r"\1", cleaned_text)
    cleaned_text = re.sub(r"\b(\w+)( \1\b)+", r"\1", cleaned_text)
    return cleaned_text.strip()

class SummaryTester:
    def __init__(self, checkpoint_path: str):
        self.config = get_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = MT5Tokenizer.from_pretrained(self.config['model_name'])

        self.model = self._load_model(checkpoint_path)
        self.model.eval()

        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def _load_model(self, checkpoint_path: str) -> CrossLingualSummarizer:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model = CrossLingualSummarizer(self.config)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        model.model.load_state_dict(state_dict, strict=False)
        return model.to(self.device)

    def generate_summary(self, text: str, lang: str) -> str:
        lang_token = f"<{lang}>"
        if not text.startswith(lang_token):
            text = f"{lang_token} {text}"

        inputs = self.tokenizer(
            text,
            max_length=self.config['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate_summary(
                inputs['input_ids'],
                inputs['attention_mask'],
                num_beams=4,
                max_length=550,
                min_length=10,
                no_repeat_ngram_size=6, 
                length_penalty=1.0
            )

        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return clean_generated_summary(generated_text)

    def evaluate_summary(self, generated: str, reference: str) -> Dict[str, Any]:
        """
        Evaluates the generated summary against a reference using ROUGE scores.
        """
        scores = self.scorer.score(reference, generated)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }


def run_tests():
    """
    Runs tests on predefined cases and displays results.
    """
    # Test data
    test_cases = [
        {   
            "lang": "al", 
            "text": "Një grua e re që humbi shikimin në moshën 12 vjeç ka ndihmuar për të përmirësuar të drejtat e njerëzve me aftësi të kufizuara. Pavarësisht sfidave që ka përjetuar, ajo ndoqi studimet e larta dhe fitoi një diplomë në psikologji. Përpjekjet e saj çuan në krijimin e një fondacioni që mbështet individët me aftësi të kufizuara, duke promovuar përfshirjen dhe aksesin në shkolla dhe vende pune. Përmes përpjekjeve të saj për mbrojtjen e të drejtave të njerëzve me aftësi të kufizuara, ajo është njohur me disa çmime, duke u bërë simbol shprese për shumë njerëz.", 
            "reference": "Një grua e re që humbi shikimin në moshën 12 vjeç u bë avokate për të drejtat e personave me aftësi të kufizuara, duke fituar një diplomë në psikologji dhe duke krijuar një fondacion për të promovuar përfshirjen dhe qasjen." 
        },
        # {
        #     "lang": "al",
        #     "text": "Një mësues në një shkollë të vogël në veri të Shqipërisë ka implementuar një metodë të re mësimdhënieje për të ndihmuar nxënësit të mësojnë më mirë. Duke përdorur teknologjinë dhe qasjen praktike, ai ka arritur të rrisë përfshirjen dhe rezultatet e nxënësve. Kjo qasje është vlerësuar nga komuniteti lokal dhe është propozuar për t'u përdorur edhe në shkolla të tjera.",
        #     "reference": "Një mësues në veri të Shqipërisë përdori teknologjinë për të rritur përfshirjen dhe rezultatet e nxënësve, duke implementuar një metodë të re mësimdhënieje."
        # },
        # {
        #     "lang": "al",
        #     "text": "Një djalë nga një fshat i vogël në Shqipëri ka krijuar një aplikacion që ndihmon fermerët të monitorojnë kushtet e tokës dhe të rrisin prodhimin. Ai filloi të punonte në këtë projekt gjatë studimeve në shkencat kompjuterike dhe tani aplikacioni është përdorur nga qindra fermerë. Ky sukses e ka bërë atë të njohur në komunitetin e teknologjisë dhe ai ka marrë disa çmime ndërkombëtare.",
        #     "reference": "Një djalë nga një fshat shqiptar krijoi një aplikacion që ndihmon fermerët të përmirësojnë prodhimin dhe fitoi njohje ndërkombëtare."
        # },
        # {
        #     "lang": "al",
        #     "text": "Një mësues në një shkollë të vogël ka zhvilluar një metodë të re mësimdhënieje për të ndihmuar nxënësit me vështirësi të të mësuarit. Ai përdori teknologjinë për të krijuar materiale interaktive që bëjnë procesin më të lehtë dhe argëtues. Kjo qasje është vlerësuar nga komuniteti arsimor dhe është propozuar për t'u aplikuar në shkolla të tjera.",
        #     "reference": "Një mësues krijoi një metodë mësimdhënieje interaktive për nxënësit me vështirësi të të mësuarit, duke përdorur teknologjinë."
        # }
    ]

    tester = SummaryTester("checkpoints/best_model_20241218_075651.pt")

    for case in test_cases:
        print(f"\nTesting {case['lang'].upper()} summarization:")
        print("-" * 50)
        print("Original text:")
        print(case['text'].strip())

        try:
            summary = tester.generate_summary(case['text'], case['lang'])
            print("\nGenerated summary:")
            print(summary)

            scores = tester.evaluate_summary(summary, case['reference'])
            print("\nROUGE scores:")
            for metric, score in scores.items():
                print(f"{metric}: {score:.4f}")

        except Exception as e:
            logger.error(f"Error processing {case['lang']} text: {str(e)}")


if __name__ == "__main__":
    run_tests()
