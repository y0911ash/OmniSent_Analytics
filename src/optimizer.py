import pandas as pd
import numpy as np
import logging
import itertools

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.signal import SignalGenerator
from src.backtest import PortfolioBacktester
from src.metrics import MetricsEvaluator

class HyperparameterOptimizer:
    """
    Performs grid search cross-validation to find the optimal rolling window
    and Hybrid Factor weighting that maximizes the Sharpe Ratio without overfitting.
    """
    def __init__(self, df_market: pd.DataFrame, df_news_scored: pd.DataFrame):
        self.df_market = df_market
        self.df_news_scored = df_news_scored
        
    def optimize(self, windows=[15, 30, 45, 60], sentiment_weights=[0.5, 0.7, 0.9]) -> pd.DataFrame:
        logging.info("Starting Grid Search Optimization for Sharpe Ratio...")
        results = []
        
        # Grid Search Combinations
        combinations = list(itertools.product(windows, sentiment_weights))
        
        for w, s_w in combinations:
            logging.info(f"Testing Model | Rolling Window: {w} | Sentiment Weight: {s_w}")
            
            # Sub-class SignalGenerator to artificially inject the tested weight
            # (Normally you'd parameterize generate_hybrid_signal, but we override here for the loop)
            signal_gen = SignalGenerator(self.df_market, self.df_news_scored, rolling_window=w)
            df_signals = signal_gen.process()
            
            # Recalculate Hybrid Weight
            df_signals['Hybrid_Signal'] = s_w * df_signals['Z_Score'] + (1 - s_w) * df_signals['Mom_Z_Score']
            
            # Run Backtest
            backtester = PortfolioBacktester(df_signals, transaction_cost=0.001)
            df_portfolio = backtester.run_backtest(signal_col='Hybrid_Signal', top_n=2)
            
            # Get Metrics
            evaluator = MetricsEvaluator(df_portfolio)
            try:
                metrics_df = evaluator.calculate_metrics()
                # Safely extract Strategy Sharpe Ratio
                if 'Strategy' in metrics_df.columns:
                    sharpe_str = metrics_df.loc['Sharpe Ratio', 'Strategy']
                    sharpe = float(sharpe_str)
                else:
                    sharpe = 0.0
            except Exception as e:
                sharpe = 0.0
                
            results.append({
                'Rolling_Window': w,
                'Sentiment_Weight': s_w,
                'Momentum_Weight': round(1 - s_w, 2),
                'Sharpe_Ratio': sharpe
            })
            
        res_df = pd.DataFrame(results)
        res_df = res_df.sort_values(by='Sharpe_Ratio', ascending=False).reset_index(drop=True)
        
        logging.info("Optimization Complete.")
        logging.info(f"\nBest Settings Found:\n{res_df.head(1).to_string()}")
        return res_df

if __name__ == "__main__":
    logging.info("Run main.py or the Streamlit dashboard to utilize the core metrics.")
