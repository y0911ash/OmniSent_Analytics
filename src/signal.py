import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SignalGenerator:
    """
    Transforms raw Sentiment and Price Data into tradable quantitative Signals.
    Handles aggregations and rolling Z-score normalizations.
    """
    def __init__(self, df_market: pd.DataFrame, df_news: pd.DataFrame, rolling_window: int = 30):
        self.df_market = df_market.copy()
        self.df_news = df_news.copy()
        self.rolling_window = rolling_window

    def aggregate_daily_sentiment(self) -> pd.DataFrame:
        """
        Group sentiment scores by Ticker and Date.
        """
        logging.info("Aggregating daily sentiment scores...")
        if 'Sentiment_Score' not in self.df_news.columns:
            raise ValueError("Sentiment_Score column is missing. Run BERT analysis first.")
            
        daily_sentiment = self.df_news.groupby(['Ticker', 'Date'])['Sentiment_Score'].mean().reset_index()
        return daily_sentiment

    def normalize_signal(self, df_merged: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates rolling Z-score to prevent lookahead bias.
        """
        logging.info(f"Applying rolling Z-score normalization (Window: {self.rolling_window} days)...")
        
        # Sort values properly to ensure rolling window doesn't look ahead or across tickers
        df_merged = df_merged.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)
        
        # Calculate Rolling Mean and Std
        df_merged['Rolling_Mean'] = df_merged.groupby('Ticker')['Sentiment_Score'].transform(
            lambda x: x.rolling(window=self.rolling_window, min_periods=5).mean()
        )
        df_merged['Rolling_Std'] = df_merged.groupby('Ticker')['Sentiment_Score'].transform(
            lambda x: x.rolling(window=self.rolling_window, min_periods=5).std()
        )
        
        # Z-Score Signal
        df_merged['Z_Score'] = (df_merged['Sentiment_Score'] - df_merged['Rolling_Mean']) / (df_merged['Rolling_Std'] + 1e-6)
        
        # Fill NaNs from the initial rolling window with 0
        df_merged['Z_Score'] = df_merged['Z_Score'].fillna(0)
        
        return df_merged

    def generate_hybrid_signal(self, df_merged: pd.DataFrame) -> pd.DataFrame:
        """
        Combines NLP Z-Score with Price Momentum for an Elite 'Flagship' Signal.
        """
        logging.info("Generating Hybrid Alpha Signal (NLP + Price Momentum)...")
        # Simple Momentum Factor: the sign of the previous 5 days' return
        df_merged['Close_Lag_5'] = df_merged.groupby('Ticker')['Close'].shift(5)
        df_merged['Momentum_Factor'] = (df_merged['Close'] - df_merged['Close_Lag_5']) / df_merged['Close_Lag_5']
        
        # Normalize Momentum to keep it purely cross-sectional or rolling
        df_merged['Mom_Rolling_Mean'] = df_merged.groupby('Ticker')['Momentum_Factor'].transform(
            lambda x: x.rolling(window=self.rolling_window, min_periods=5).mean()
        )
        df_merged['Mom_Rolling_Std'] = df_merged.groupby('Ticker')['Momentum_Factor'].transform(
            lambda x: x.rolling(window=self.rolling_window, min_periods=5).std()
        )
        
        df_merged['Mom_Z_Score'] = (df_merged['Momentum_Factor'] - df_merged['Mom_Rolling_Mean']) / (df_merged['Mom_Rolling_Std'] + 1e-6)
        df_merged['Mom_Z_Score'] = df_merged['Mom_Z_Score'].fillna(0)
        
        # Hybrid Signal (Equal weight, can be empirically tuned)
        df_merged['Hybrid_Signal'] = 0.7 * df_merged['Z_Score'] + 0.3 * df_merged['Mom_Z_Score']
        
        return df_merged

    def process(self) -> pd.DataFrame:
        """
        Main runner: aggregates, merges, and generates signals.
        """
        daily_sentiment = self.aggregate_daily_sentiment()
        
        # Merge market data with aligned daily sentiment
        # Note: df_market and daily_sentiment share 'Ticker' and 'Date'.
        df_merged = pd.merge(self.df_market, daily_sentiment, on=['Ticker', 'Date'], how='left')
        
        # Fill days with no news with 0 sentiment
        df_merged['Sentiment_Score'] = df_merged['Sentiment_Score'].fillna(0)
        
        df_merged = self.normalize_signal(df_merged)
        df_merged = self.generate_hybrid_signal(df_merged)
        
        return df_merged

if __name__ == "__main__":
    logging.info("Run main.py for full end-to-end Signal processing.")
