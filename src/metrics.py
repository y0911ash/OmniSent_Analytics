import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MetricsEvaluator:
    """
    Computes institutional-grade quantitative metrics and generates Tear Sheets.
    """
    def __init__(self, portfolio_df: pd.DataFrame, risk_free_rate: float = 0.02):
        self.df = portfolio_df.copy()
        self.rf = risk_free_rate / 252 # Daily risk-free rate
        
    def calculate_metrics(self):
        """Calculates core quant metrics for the strategy vs benchmark."""
        metrics = {}
        
        for prefix, col in [('Strategy', 'Strategy_Return'), ('Benchmark', 'Benchmark_Return')]:
            returns = self.df[col].dropna()
            if len(returns) == 0:
                continue
                
            # Annualized Return
            ann_ret = (1 + returns.mean()) ** 252 - 1
            
            # Annualized Volatility
            ann_vol = returns.std() * np.sqrt(252)
            
            # Sharpe Ratio
            excess_ret = returns - self.rf
            sharpe = (excess_ret.mean() / returns.std()) * np.sqrt(252)
            
            # Max Drawdown
            cum_ret = (1 + returns).cumprod()
            rolling_max = cum_ret.cummax()
            drawdown = (cum_ret - rolling_max) / rolling_max
            max_dd = drawdown.min()
            
            # Hit Ratio (% of positive days)
            hit_ratio = (returns > 0).mean()
            
            metrics[prefix] = {
                'Ann. Return': f"{ann_ret * 100:.2f}%",
                'Ann. Volatility': f"{ann_vol * 100:.2f}%",
                'Sharpe Ratio': f"{sharpe:.2f}",
                'Max Drawdown': f"{max_dd * 100:.2f}%",
                'Hit Ratio': f"{hit_ratio * 100:.2f}%"
            }
            
        return pd.DataFrame(metrics)

    def calculate_ic(self, signals_df: pd.DataFrame, signal_col: str = 'Hybrid_Signal') -> float:
        """
        Calculates the Information Coefficient (Spearman Rank Correlation between Signal and Forward Return).
        """
        clean_df = signals_df.dropna(subset=[signal_col, 'Forward_Return'])
        # Cross-sectional IC per day, then averaged
        ic_series = clean_df.groupby('Date').apply(
            lambda x: x[signal_col].corr(x['Forward_Return'], method='spearman')
        )
        mean_ic = ic_series.mean()
        logging.info(f"Mean Information Coefficient (IC): {mean_ic:.4f}")
        return mean_ic

    def plot_tear_sheet(self, save_path: str = "tear_sheet.png"):
        """Generates a visual performance Tear Sheet."""
        logging.info("Generating Tear Sheet visualization...")
        
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot 1: Cumulative Returns
        ax1.plot(self.df['Date'], self.df['Cumulative_Return'], label='Hybrid BERT Strategy', color='#00ff00', linewidth=2)
        ax1.plot(self.df['Date'], self.df['Benchmark_Cum_Return'], label='Equal-Weight Benchmark', color='#ffffff', alpha=0.6, linestyle='--')
        
        ax1.set_title('Cumulative Portfolio Returns', fontsize=16)
        ax1.set_ylabel('Growth of $1')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.2)
        
        # Plot 2: Drawdown
        strat_cum = self.df['Cumulative_Return']
        strat_drawdown = (strat_cum - strat_cum.cummax()) / strat_cum.cummax()
        ax2.fill_between(self.df['Date'], strat_drawdown, 0, color='#ff0000', alpha=0.5)
        ax2.set_title('Strategy Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        logging.info(f"Tear Sheet saved to {save_path}")
