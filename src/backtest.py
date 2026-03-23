import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PortfolioBacktester:
    """
    Simulates a quantitative Long/Short strategy based on Alpha Signals.
    Supports transaction costs and multi-stock daily ranking.
    """
    def __init__(self, df_signals: pd.DataFrame, transaction_cost: float = 0.001):
        self.df = df_signals.copy()
        self.transaction_cost = transaction_cost

    def run_backtest(self, signal_col: str = 'Hybrid_Signal', top_n: int = 2) -> pd.DataFrame:
        """
        Runs cross-sectional ranking per day.
        Buys Top N stocks (Long), Sells Bottom N stocks (Short).
        Returns a time series dataframe of Portfolio Returns.
        """
        logging.info(f"Running Backtest using {signal_col}. Long/Short Top/Bottom {top_n}...")
        
        # Ensure we have required columns
        required_cols = ['Date', 'Ticker', 'Forward_Return', signal_col]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing expected column: {col}")

        # Drop NaNs
        clean_df = self.df.dropna(subset=[signal_col, 'Forward_Return'])

        # Group by Date to rank cross-sectionally
        portfolio_returns = []
        
        for date, group in clean_df.groupby('Date'):
            num_stocks = len(group)
            
            if num_stocks < top_n * 2:
                # Not enough stocks to form Long and Short leg, hold cash
                portfolio_returns.append({'Date': date, 'Strategy_Return': 0.0, 'Turnover': 0.0})
                continue
            
            # Rank stocks natively
            ranked = group.sort_values(by=signal_col, ascending=False)
            
            # Long Leg: Top N highest signals (1/N equal weight)
            long_leg = ranked.head(top_n)
            long_return = long_leg['Forward_Return'].mean()
            
            # Short Leg: Bottom N lowest signals (-1/N equal weight)
            short_leg = ranked.tail(top_n)
            short_return = short_leg['Forward_Return'].mean() * -1 # Multiply by -1 since we are shorting
            
            # Total strategy return (Assuming 100% Gross Exposure: 50% Long, 50% Short)
            daily_gross_return = (long_return + short_return) / 2
            
            # Add transaction cost as a simplification (Assuming 100% turnover daily for POC)
            daily_net_return = daily_gross_return - (self.transaction_cost * 2)
            
            portfolio_returns.append({
                'Date': date,
                'Strategy_Return': daily_net_return
            })

        portfolio_df = pd.DataFrame(portfolio_returns)
        portfolio_df = portfolio_df.sort_values('Date').reset_index(drop=True)
        
        # Calculate cumulative returns
        portfolio_df['Cumulative_Return'] = (1 + portfolio_df['Strategy_Return']).cumprod()
        
        # Calculate Benchmark (Equal weight average of all available stocks for that day acting as 'Market')
        benchmark_returns = clean_df.groupby('Date')['Forward_Return'].mean().reset_index()
        benchmark_returns.rename(columns={'Forward_Return': 'Benchmark_Return'}, inplace=True)
        portfolio_df = pd.merge(portfolio_df, benchmark_returns, on='Date', how='left')
        portfolio_df['Benchmark_Cum_Return'] = (1 + portfolio_df['Benchmark_Return']).cumprod()
        
        return portfolio_df

if __name__ == "__main__":
    logging.info("Run main.py to execute the full backtest pipeline.")
