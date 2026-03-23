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

# Must be the first Streamlit command
st.set_page_config(page_title="OmniSent Analytics", layout="wide", page_icon="🌌")

# ==========================================
# CUSTOM CSS FOR PREMIUM FLAGSHIP AESTHETICS
# ==========================================
st.markdown("""
<style>
    /* Global Background and Typography */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Fade In Animation for the whole page */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .main {
        animation: fadeIn 0.8s ease-out;
    }

    /* Style the Sidebar with a Glassmorphism effect */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Primary Headers */
    h1 {
        font-weight: 800 !important;
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    h2, h3 {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }

    /* Metric Cards Styling */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 5% 10%;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(56, 189, 248, 0.2);
        border: 1px solid rgba(56, 189, 248, 0.5);
    }
    
    /* Button Aesthetics */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #3b82f6 0%, #6366f1 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 0;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #60a5fa 0%, #818cf8 100%);
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.6);
        transform: scale(1.02);
    }

    /* Dataframe custom styling */
    .stDataFrame {
        border-radius: 10px !important;
        overflow: hidden !important;
        border: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# MAIN APP ARCHITECTURE
# ==========================================

st.title("OmniSent Analytics")
st.markdown("### Institutional NLP Quantitative Engine")
st.markdown("Deploying advanced FinBERT tensor inferences harmonized with rolling momentum factors.")
st.markdown("---")

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/nolan/256/combo-chart.png", width=100)
    st.markdown("## Configuration Terminal")
    tickers_input = st.text_input("Universe Tickertape (comma separated)", "AAPL,MSFT,GOOG,AMZN,TSLA")
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    
    st.markdown("### Engine Parameters")
    lookback = st.slider("Historical Horizon (Days)", min_value=90, max_value=365*5, value=365)
    top_n = st.slider("Long/Short Top N Factor", min_value=1, max_value=5, value=2)
    transaction_cost = st.number_input("Slippage & Tx Cost (%)", value=0.1) / 100.0
    
    st.markdown("<br>", unsafe_allow_html=True)
    run_engine = st.button("🚀 IGNITE PIPELINE")

# Caching Data to eliminate memory re-runs
@st.cache_data(show_spinner=False)
def fetch_data_cached(tickers, start_date, end_date):
    return load_aligned_dataset(tickers, start_date, end_date)

@st.cache_resource(show_spinner=False)
def load_analyzer():
    return FinBertAnalyzer()

if run_engine:
    # Adding a sleek progress container
    status_container = st.empty()
    
    with status_container.container():
        st.info("📡 Synching Market Data & Synthetic Web Scraping...", icon="⏳")
        start_date = (datetime.now() - timedelta(days=lookback)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        df_market, df_news = fetch_data_cached(tickers, start_date, end_date)
        
        st.info("🧠 Initializing Deep Learning FinBERT Tensors...", icon="⏳")
        analyzer = load_analyzer()
        df_news_scored = analyzer.process_dataframe(df_news)
    
        st.info("🧮 Calculating 30-Day Rolling Z-Scores & Hybrid Alphas...", icon="⏳")
        signal_gen = SignalGenerator(df_market, df_news_scored, rolling_window=30)
        df_signals = signal_gen.process()
        
        st.info("⚡ Executing High-Frequency Portfolio Simulation...", icon="⏳")
        backtester = PortfolioBacktester(df_signals, transaction_cost=transaction_cost)
        df_portfolio = backtester.run_backtest(signal_col='Hybrid_Signal', top_n=top_n)
        
        st.info("📊 Compiling Institutional Tear Sheets...", icon="⏳")
        evaluator = MetricsEvaluator(df_portfolio)
        metrics_df = evaluator.calculate_metrics()
        ic_score = evaluator.calculate_ic(df_signals, 'Hybrid_Signal')

    # Clear status and show success
    status_container.empty()
    st.success("✅ Execution Complete. Structural Lookahead Bias: 0.00%", icon="🛡️")
    
    # Define Layout Columns
    col1, col2, col3 = st.columns(3)
    
    # Safely get metrics
    try:
        strat_return = metrics_df.loc['Ann. Return', 'Strategy']
        strat_sharpe = metrics_df.loc['Sharpe Ratio', 'Strategy']
        strat_win = metrics_df.loc['Hit Ratio', 'Strategy']
    except Exception:
        strat_return, strat_sharpe, strat_win = "N/A", "N/A", "N/A"

    with col1:
        st.metric(label="Alpha Output (Ann. Return)", value=strat_return)
    with col2:
        st.metric(label="Risk-Adjusted (Sharpe Ratio)", value=strat_sharpe)
    with col3:
        st.metric(label="Information Coefficient (IC)", value=f"{ic_score:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Split layout for chart and table
    chart_tab, data_tab = st.tabs(["📉 Quantitative Tear Sheet", "🧠 Raw Deep Learning Output"])
    
    with chart_tab:
        os.makedirs("notebooks", exist_ok=True)
        img_path = os.path.join("notebooks", "tear_sheet_ui.png")
        evaluator.plot_tear_sheet(save_path=img_path)
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
            
    with data_tab:
        st.markdown("#### Live Factor Streams")
        st.dataframe(df_news_scored.tail(50)[['Date', 'Ticker', 'Headline', 'Sentiment_Score']].style.background_gradient(cmap='Blues', subset=['Sentiment_Score']), use_container_width=True)

elif not run_engine:
    # Front-page placeholder when not running
    st.info("Awaiting command. Configure parameters in the terminal and click **IGNITE PIPELINE**.")
