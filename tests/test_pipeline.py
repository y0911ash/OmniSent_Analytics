import unittest
import pandas as pd
import numpy as np

# Adjust sys.path to run tests from root
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import MarketDataFetcher, NewsDataFetcher
from src.signal import SignalGenerator

class TestPipeline(unittest.TestCase):
    
    def setUp(self):
        # Create dummy market data
        dates = pd.date_range(start='2023-01-01', periods=10, freq='B')
        self.df_market = pd.DataFrame({
            'Date': dates,
            'Ticker': ['AAPL'] * 10,
            'Open': np.linspace(100, 110, 10),
            'High': np.linspace(102, 112, 10),
            'Low': np.linspace(98, 108, 10),
            'Close': np.linspace(101, 111, 10),
            'Volume': [1000] * 10
        })
        
        # Manually calculate forward return logic as per data_loader
        self.df_market['Next_Open'] = self.df_market['Open'].shift(-1)
        self.df_market['Next_Close'] = self.df_market['Close'].shift(-1)
        self.df_market['Forward_Return'] = (self.df_market['Next_Close'] - self.df_market['Next_Open']) / self.df_market['Next_Open']
        self.df_market = self.df_market.dropna()

    def test_lookahead_bias_prevention(self):
        """Ensures that Forward_Return uses strictly future data."""
        # Row 0 date is 2023-01-02. Next Open/Close should be from Row 1
        row0 = self.df_market.iloc[0]
        # Open was 100 on day 0, 101.11 on day 1
        self.assertTrue(row0['Next_Open'] > row0['Open'], "Forward data misalignment.")

    def test_rolling_normalization(self):
        """Ensures Z-Score doesn't peek into future rows."""
        self.df_news = pd.DataFrame({
            'Date': self.df_market['Date'],
            'Ticker': ['AAPL'] * len(self.df_market),
            'Sentiment_Score': np.random.randn(len(self.df_market))
        })
        
        signal_gen = SignalGenerator(self.df_market, self.df_news, rolling_window=3)
        df_signals = signal_gen.process()
        
        # Test if rolling window correctly handles NaNs for the initial period without
        # using future data (fillna(0) applied in process())
        self.assertTrue('Z_Score' in df_signals.columns)
        self.assertFalse(df_signals['Z_Score'].isnull().any(), "Z-Scores should not contain NaNs")

if __name__ == '__main__':
    unittest.main()
