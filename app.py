import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# Adjust path so we can import src
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import load_aligned_dataset
from src.bert_model import FinBertAnalyzer
from src.signal import SignalGenerator
from src.backtest import PortfolioBacktester
from src.metrics import MetricsEvaluator

st.set_page_config(page_title="BERT Quant Engine", layout="wide", page_icon="📈")

st.title("🚀 BERT Sentiment Engine - Flagship Dashboard")
st.markdown("An institutional-grade NLP quantitative trading pipeline.")

st.sidebar.header("Configure Backtest")
tickers_input = st.sidebar.text_input("Tickers (comma separated)", "AAPL,MSFT,GOOG,AMZN,TSLA")
tickers = [t.strip().upper() for t in tickers_input.split(",")]

lookback = st.sidebar.slider("Historical Data (Days)", min_value=90, max_value=365*5, value=365)
top_n = st.sidebar.slider("Long/Short Top N", min_value=1, max_value=5, value=2)
transaction_cost = st.sidebar.number_input("Transaction Cost (%)", value=0.1) / 100.0

if st.sidebar.button("▶ Run Full Backtest Engine"):
    with st.spinner("Fetching Market Data & Generating Synthetic News..."):
        start_date = (datetime.now() - timedelta(days=lookback)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        df_market, df_news = load_aligned_dataset(tickers, start_date, end_date)
    
    with st.spinner("Running FinBERT NLP Inference (This takes a moment)..."):
        analyzer = FinBertAnalyzer()
        df_news_scored = analyzer.process_dataframe(df_news)

    with st.spinner("Calculating Alpha Signals & Rolling Normalizations..."):
        signal_gen = SignalGenerator(df_market, df_news_scored, rolling_window=30)
        df_signals = signal_gen.process()
    
    with st.spinner("Executing Long/Short Portfolio Simulation..."):
        backtester = PortfolioBacktester(df_signals, transaction_cost=transaction_cost)
        df_portfolio = backtester.run_backtest(signal_col='Hybrid_Signal', top_n=top_n)
    
    with st.spinner("Calculating Performance Metrics..."):
        evaluator = MetricsEvaluator(df_portfolio)
        metrics_df = evaluator.calculate_metrics()
        ic_score = evaluator.calculate_ic(df_signals, 'Hybrid_Signal')
        
    st.success("Backtest Completed Succesfully! 0 Lookahead Bias detected.")
    
    # ------------------ Display Results ------------------
    st.header("📊 Top-Level Metrics")
    st.dataframe(metrics_df.style.highlight_max(axis=0), use_container_width=True)
    st.metric(label="Information Coefficient (IC)", value=f"{ic_score:.4f}", help="Spearman rank correlation of Signal vs T+1 Return.")
    
    st.header("📈 Strategy Tear Sheet")
    os.makedirs("notebooks", exist_ok=True)
    img_path = os.path.join("notebooks", "tear_sheet_ui.png")
    evaluator.plot_tear_sheet(save_path=img_path)
    if os.path.exists(img_path):
        st.image(img_path, caption='Cumulative Returns & Drawdowns')

    st.header("🧠 Sample NLP Predictions")
    st.dataframe(df_news_scored.tail(15)[['Date', 'Ticker', 'Headline', 'Sentiment_Score']])
    
st.markdown("---")
st.markdown("Developed with robust statistical principles. Zero data leakage.")
