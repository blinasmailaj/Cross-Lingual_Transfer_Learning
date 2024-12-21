import torch
from transformers import MT5Tokenizer
from models.summarizer import CrossLingualSummarizer
from config import get_config
import logging
import os
from rouge_score import rouge_scorer
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SummaryTester:
    def __init__(self, checkpoint_path: str):
        self.config = get_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = MT5Tokenizer.from_pretrained(self.config['model_name'])
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        # Initialize ROUGE scorer
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
        # Add language token if not present
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
                no_repeat_ngram_size=5,
                length_penalty=1.0
            )
            
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def evaluate_summary(self, generated: str, reference: str) -> Dict[str, Any]:
        scores = self.scorer.score(reference, generated)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }

def run_tests():
    # Test data
    test_cases = [
    {   "lang": "al", 
        "text": "Një grua e re që humbi shikimin në moshën 12 vjeç ka ndihmuar për të përmirësuar të drejtat e njerëzve me aftësi të kufizuara. Pavarësisht sfidave që ka përjetuar, ajo ndoqi studimet e larta dhe fitoi një diplomë në psikologji. Përpjekjet e saj çuan në krijimin e një fondacioni që mbështet individët me aftësi të kufizuara, duke promovuar përfshirjen dhe aksesin në shkolla dhe vende pune. Përmes përpjekjeve të saj për mbrojtjen e të drejtave të njerëzve me aftësi të kufizuara, ajo është njohur me disa çmime, duke u bërë simbol shprese për shumë njerëz.", 
        "reference": "Një grua e re që humbi shikimin në moshën 12 vjeç u bë avokate për të drejtat e personave me aftësi të kufizuara, duke fituar një diplomë në psikologji dhe duke krijuar një fondacion për të promovuar përfshirjen dhe qasjen." 
    },
    {
        "lang": "al",
        "text": "Një grup studentësh nga Shqipëria ka fituar një konkurs ndërkombëtar të robotikës. Ata krijuan një robot që mund të ndihmojë në operacionet e shpëtimit pas katastrofave natyrore. Roboti është i pajisur me sensorë të avancuar dhe mund të lëvizë në terrene të vështira. Ky sukses është një dëshmi e aftësive dhe inovacionit të të rinjve shqiptarë në fushën e teknologjisë.",
        "reference": "Një grup studentësh shqiptarë ka fituar një konkurs ndërkombëtar të robotikës me një robot që ndihmon në operacionet e shpëtimit pas katastrofave natyrore, duke dëshmuar aftësitë dhe inovacionin e tyre."
    },
    {
        "lang": "al",
        "text": "Një artist i ri shqiptar ka fituar një çmim ndërkombëtar për pikturat e tij. Ai është njohur për stilin e tij unik dhe përdorimin e ngjyrave të gjalla. Pikturat e tij janë ekspozuar në galeri të ndryshme në mbarë botën dhe kanë marrë vlerësime të larta nga kritikët. Artisti shpreson të frymëzojë të rinjtë e tjerë për të ndjekur pasionet e tyre artistike.",
        "reference": "Një artist shqiptar ka fituar një çmim ndërkombëtar për pikturat e tij, të njohura për stilin unik dhe ngjyrat e gjalla, dhe ka ekspozuar në galeri të ndryshme në mbarë botën."
    },
    {
        "lang": "al",
        "text": "Një ekip shkencëtarësh ka zbuluar një metodë të re për të luftuar ndryshimet klimatike. Ata kanë zhvilluar një teknologji që kap dhe ruan dioksidin e karbonit nga atmosfera. Kjo teknologji mund të ndihmojë në reduktimin e emetimeve të gazrave serë dhe të ngadalësojë ngrohjen globale. Shkencëtarët shpresojnë se kjo metodë do të adoptohet gjerësisht në të ardhmen për të mbrojtur planetin tonë.",
        "reference": "Një ekip shkencëtarësh ka zhvilluar një teknologji për të kapur dhe ruajtur dioksidin e karbonit nga atmosfera, duke ndihmuar në luftën kundër ndryshimeve klimatike dhe ngrohjes globale."
    }
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



# Input Text: Should contain sufficient detail and context to ensure a meaningful summary.
# Reference Summary: Should capture key ideas, removing non-essential details.
# The goal is to compare the generated summary with the reference summary and calculate ROUGE scores to measure performance.