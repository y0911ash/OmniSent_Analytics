import os
import logging
from datetime import datetime, timedelta
import pandas as pd

# Import our custom modules
from src.data_loader import load_aligned_dataset
from src.bert_model import FinBertAnalyzer
from src.signal import SignalGenerator
from src.backtest import PortfolioBacktester
from src.metrics import MetricsEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("="*60)
    print("🚀 BERT Sentiment Engine — Flagship Quant Pipeline")
    print("="*60)

    # 1. Project Setup
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # 2. Data Layer
    print("\n[1/5] Executing Data Layer (Fetching Market & Simulating News)...")
    df_market, df_news = load_aligned_dataset(tickers, start_date, end_date)
    
    # 3. NLP Layer (FinBERT)
    print("\n[2/5] Executing NLP Layer (FinBERT Financial Sentiment Analysis)...")
    analyzer = FinBertAnalyzer()
    df_news_scored = analyzer.process_dataframe(df_news)
    
    # 4. Signal Generation Layer
    print("\n[3/5] Executing Signal Generation (Rolling Z-Score & Hybrid Factors)...")
    signal_gen = SignalGenerator(df_market, df_news_scored, rolling_window=30)
    df_signals = signal_gen.process()
    
    # 5. Backtesting Layer
    print("\n[4/5] Executing Institutional Backtesting Engine...")
    backtester = PortfolioBacktester(df_signals, transaction_cost=0.001)
    df_portfolio = backtester.run_backtest(signal_col='Hybrid_Signal', top_n=2)
    
    # 6. Evaluation Layer
    print("\n[5/5] Executing Strategy Evaluation & Benchmarking...")
    evaluator = MetricsEvaluator(df_portfolio)
    metrics_table = evaluator.calculate_metrics()
    
    # Also calculate the Information Coefficient (IC) using the original signals across all stocks
    evaluator.calculate_ic(df_signals, 'Hybrid_Signal')
    
    # Print Metrics
    print("\n" + "="*45)
    print("📊 Strategy vs Benchmark Performance Metrics")
    print("="*45)
    print(metrics_table.to_string())
    print("="*45)
    
    # Generate Visual Tear Sheet
    os.makedirs('notebooks', exist_ok=True)
    tear_sheet_path = os.path.join('notebooks', 'tear_sheet.png')
    evaluator.plot_tear_sheet(save_path=tear_sheet_path)
    
    print(f"\n✅ Pipeline Complete! Tear sheet saved to {tear_sheet_path}")

if __name__ == "__main__":
    main()
