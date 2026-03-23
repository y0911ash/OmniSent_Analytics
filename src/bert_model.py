import torch
from transformers import BertTokenizer, BertForSequenceClassification
import logging
from tqdm import tqdm
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FinBertAnalyzer:
    """
    NLP Layer: Uses ProsusAI/finbert to calculate financial sentiment scores.
    """
    def __init__(self, model_name="ProsusAI/finbert"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Loading {model_name} onto {self.device}...")
        
        # Load pre-trained model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # FinBERT labels: 0=positive, 1=negative, 2=neutral (ProsusAI specific mapping)
        # Note: We extract the raw logits, apply softmax to get probabilities.
    
    def get_sentiment_score(self, text: str) -> float:
        """
        Calculates sentiment score for a single string.
        Score = P(Positive) - P(Negative)
        """
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # ProsusAI/finbert specific label indexing:
            # probs[0][0] -> positive
            # probs[0][1] -> negative
            # probs[0][2] -> neutral
            pos_prob = probs[0][0].item()
            neg_prob = probs[0][1].item()
            
            return pos_prob - neg_prob

    def process_dataframe(self, df: pd.DataFrame, text_column="Headline") -> pd.DataFrame:
        """
        Batch processes a dataframe and appends the Sentiment_Score column.
        """
        logging.info(f"Analyzing {len(df)} headlines with FinBERT...")
        # To avoid overhead for very large sets, a PyTorch DataLoader is ideal, 
        # but for POC, iterative processing with tqdm is fine.
        scores = []
        for text in tqdm(df[text_column], desc="Scoring Sentiments"):
            scores.append(self.get_sentiment_score(text))
            
        df['Sentiment_Score'] = scores
        return df

if __name__ == "__main__":
    # Test the analyzer
    analyzer = FinBertAnalyzer()
    texts = [
        "Apple beats earnings expectations, surging 5% in after-hours trading.",
        "Inflation concerns rise as supply chain bottlenecks cause production delays.",
        "The company announced its standard quarterly dividend."
    ]
    df_test = pd.DataFrame({"Headline": texts})
    df_test = analyzer.process_dataframe(df_test)
    print(df_test)
