# OmniSent Analytics 🚀
An institutional-grade NLP quantitative trading pipeline utilizing FinBERT sequence classification, rolling feature momentum factors, and rigorous Z-score normalizations to mathematically extract cross-sectional alpha from financial news.

## Core Architecture
- **NLP Layer:** `ProsusAI/finbert` (HuggingFace) mapping $P(\text{Positive}) - P(\text{Negative})$ sentiment points safely against daily asset returns.
- **Factor Generation:** Rolling 30-day Z-scores blended cross-sectionally to rigorously eliminate Lookahead Bias.
- **Interactive UI:** A real-time `Streamlit` engine to generate Quant Tear-Sheets and test backtest parameters interactively.
- **Hyperparameter Tuning:** Automated Grid-Search logic over signal smoothing models.

## How to Run locally:
1. Ensure your Python environment is set up.
2. `pip install -r requirements.txt`
3. Launch the dashboard to interact with the backtests:
```bash
streamlit run app.py
```
Or run the pure terminal pipeline:
```bash
python main.py
```
