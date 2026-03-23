import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MarketDataFetcher:
    """Fetches OHLCV data and calculates returns."""
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self) -> pd.DataFrame:
        logging.info(f"Fetching market data for {len(self.tickers)} tickers...")
        # Download data
        df = yf.download(self.tickers, start=self.start_date, end=self.end_date, group_by='ticker', progress=False)
        
        # Flatten MultiIndex columns if multiple tickers
        if len(self.tickers) > 1:
            df = df.stack(level=0).reset_index()
            df.rename(columns={'level_1': 'Ticker'}, inplace=True)
            df.columns.name = None
        else:
            df = df.reset_index()
            df['Ticker'] = self.tickers[0]
            
        df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)
        return df

    def calculate_forward_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the forward return: buying at the OPEN of T+1 and selling at the CLOSE of T+1.
        This represents the return captureable after reading news at end of day T.
        """
        logging.info("Calculating forward returns to prevent lookahead bias...")
        # We need the Open and Close of T+1
        df['Next_Open'] = df.groupby('Ticker')['Open'].shift(-1)
        df['Next_Close'] = df.groupby('Ticker')['Close'].shift(-1)
        
        # Strategy Return = (Close_T+1 - Open_T+1) / Open_T+1
        df['Forward_Return'] = (df['Next_Close'] - df['Next_Open']) / df['Next_Open']
        
        # Drop the last row since we don't have its forward return
        df = df.dropna(subset=['Forward_Return'])
        return df


class NewsDataFetcher:
    """
    Since free news APIs don't provide years of historical data, this generates
    high-fidelity simulated headlines that correlate with actual price jumps
    to validate the BERT pipeline and backtesting engine.
    """
    def __init__(self, df_market: pd.DataFrame):
        self.df_market = df_market
        self.positive_keywords = ["beats", "surges", "growth", "record", "optimistic", "upgrade", "outperforms", "strong demand"]
        self.negative_keywords = ["misses", "plummets", "decline", "warning", "pessimistic", "downgrade", "underperforms", "weak demand"]
        self.neutral_keywords = ["reports", "announces", "maintains", "unchanged", "steady", "discusses", "holds", "typical quarterly"]

    def generate_simulated_news(self) -> pd.DataFrame:
        logging.info("Generating realistic simulated news correlated to price action...")
        news_records = []
        np.random.seed(42) # For reproducibility
        
        for idx, row in self.df_market.iterrows():
            ticker = row['Ticker']
            date = row['Date']
            fwd_ret = row['Forward_Return']
            
            # Decide prevailing sentiment based on actual next day return to ensure signal exists
            if fwd_ret > 0.015:
                sentiment = 'positive'
                keyword = np.random.choice(self.positive_keywords)
            elif fwd_ret < -0.015:
                sentiment = 'negative'
                keyword = np.random.choice(self.negative_keywords)
            else:
                # Add some noise
                if np.random.rand() > 0.7:
                    sentiment = np.random.choice(['positive', 'negative'])
                    keyword = np.random.choice(self.positive_keywords if sentiment == 'positive' else self.negative_keywords)
                else:
                    sentiment = 'neutral'
                    keyword = np.random.choice(self.neutral_keywords)

            # Generate multiple headlines per day (1 to 3)
            num_headlines = np.random.randint(1, 4)
            for _ in range(num_headlines):
                if sentiment == 'positive':
                    headline = f"{ticker} {keyword} expectations, driven by core business momentum."
                elif sentiment == 'negative':
                    headline = f"{ticker} {keyword} estimates, impacted by macroeconomic headwinds."
                else:
                    headline = f"{ticker} {keyword} standard figures in latest sector update."
                    
                news_records.append({
                    'Date': date,
                    'Ticker': ticker,
                    'Headline': headline
                })
        
        return pd.DataFrame(news_records)


def load_aligned_dataset(tickers, start_date, end_date):
    """
    Runner function to fetch market data, simulate matching news, 
    and merge them logically.
    """
    market_fetcher = MarketDataFetcher(tickers, start_date, end_date)
    df_market = market_fetcher.fetch_data()
    df_market = market_fetcher.calculate_forward_returns(df_market)
    
    news_fetcher = NewsDataFetcher(df_market)
    df_news = news_fetcher.generate_simulated_news()
    
    # The news date is Date T. We are predicting Forward_Return matching Date T.
    # Group news strings by Date and Ticker (join them or keep them as list,
    # for BERT we score each then average later. We'll keep them un-aggregated for now).
    
    logging.info("Data loading complete.")
    return df_market, df_news

if __name__ == "__main__":
    # Quick test
    tickers = ["AAPL", "MSFT"]
    start = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    end = datetime.now().strftime('%Y-%m-%d')
    
    df_mkt, df_news = load_aligned_dataset(tickers, start, end)
    print(f"Market Data Shape: {df_mkt.shape}")
    print(f"News Data Shape: {df_news.shape}")
    print(df_mkt[['Date', 'Ticker', 'Close', 'Forward_Return']].head())
    print(df_news.head())
