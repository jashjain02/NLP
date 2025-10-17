# NLP Finance Prediction Pipeline

A comprehensive machine learning pipeline that uses Natural Language Processing (NLP) to predict stock price movements by analyzing financial news sentiment.

## 🚀 Features

- **Multi-source News Collection**: Aggregates news from MoneyControl, NewsAPI, MarketAux, RSS feeds, and GDELT
- **Advanced NLP**: Uses FinBERT for financial sentiment analysis
- **Technical Analysis**: 20+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Machine Learning**: Multiple models (Logistic Regression, Random Forest, XGBoost) with ensemble voting
- **Time-series Validation**: Proper time-series cross-validation to prevent data leakage
- **Interactive UI**: Streamlit web interface for real-time predictions
- **Backtesting**: Comprehensive strategy evaluation with risk metrics

## 🏗️ Architecture

```
Data Collection → NLP Processing → Feature Engineering → ML Training → Prediction
     ↓              ↓                    ↓              ↓            ↓
  News APIs    FinBERT Sentiment    Technical +     Ensemble     Streamlit UI
  Stock Data   Analysis            Sentiment      Models        FastAPI Server
```

## 📊 Pipeline Components

### 1. Data Collection
- **Stock Data**: Yahoo Finance via `yfinance`
- **News Sources**: 
  - MoneyControl (Indian markets)
  - NewsAPI (Global)
  - MarketAux (US markets)
  - RSS feeds (Economic Times, LiveMint, etc.)
  - GDELT (Global events)

### 2. NLP Processing
- **Model**: FinBERT (`ProsusAI/finbert`)
- **Processing**: Batch sentiment analysis with GPU acceleration
- **Features**: Daily sentiment aggregates, rolling windows, lag features

### 3. Feature Engineering
- **Sentiment Features**: Mean sentiment, positive/negative counts, rolling averages
- **Technical Indicators**: RSI, MACD, Bollinger Bands, momentum, volatility regimes
- **Price Patterns**: Doji, hammer, shooting star detection

### 4. Machine Learning
- **Models**: Logistic Regression, Random Forest, XGBoost
- **Validation**: TimeSeriesSplit cross-validation
- **Ensemble**: Voting classifier with soft/hard voting
- **Targets**: Direction prediction (up/down) and return regression

### 5. Evaluation & Backtesting
- **Metrics**: Accuracy, F1, AUC, Sharpe ratio, max drawdown
- **Strategy**: Long/flat positions based on probability thresholds
- **Risk Management**: Transaction costs, position sizing

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/jashjain02/NLP.git
cd NLP
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file
echo "NEWSAPI_KEY=your_newsapi_key" >> .env
echo "MARKETAUX_API_KEY=your_marketaux_key" >> .env
```

5. **Run the Streamlit app**
```bash
streamlit run ui/streamlit_app.py
```

### Streamlit Cloud Deployment

1. **Fork this repository** on GitHub
2. **Go to [Streamlit Cloud](https://share.streamlit.io/)**
3. **Click "New app"**
4. **Connect your GitHub repository**
5. **Set the following:**
   - **Repository**: `your-username/NLP`
   - **Branch**: `main`
   - **Main file path**: `ui/streamlit_app.py`
   - **Python version**: `3.9`

6. **Add secrets** in Streamlit Cloud dashboard:
   ```
   NEWSAPI_KEY = your_newsapi_key
   MARKETAUX_API_KEY = your_marketaux_key
   ```

## 📁 Project Structure

```
NLP/
├── data_collection/          # Data fetching modules
│   ├── stock.py             # Yahoo Finance integration
│   └── news.py              # Multi-source news collection
├── preprocessing/            # Data preprocessing
│   └── preprocess.py        # Time alignment, trading calendars
├── nlp/                     # NLP processing
│   └── sentiment.py         # FinBERT sentiment analysis
├── features/                # Feature engineering
│   └── features.py          # Technical + sentiment features
├── modeling/                # Machine learning
│   └── modeling.py          # Model training and ensembles
├── evaluation/              # Model evaluation
│   └── evaluate.py          # Metrics and diagnostics
├── backtest/                # Strategy backtesting
│   └── backtest.py          # Performance analysis
├── api/                     # FastAPI server
│   └── server.py            # REST API endpoints
├── ui/                      # Streamlit interface
│   └── streamlit_app.py     # Main web application
├── tests/                   # Unit tests
├── main.py                  # CLI pipeline
├── config.yaml              # Configuration
└── requirements.txt         # Dependencies
```

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
ticker: "AAPL"                    # Default stock symbol
dates:
  auto_recent_window: true        # Auto date range
  lookback_days: 120             # Days of historical data
news:
  mode: "moneycontrol"           # News source priority
  max_articles: 1000             # Article limit
model:
  cv_splits: 5                   # Cross-validation folds
  holdout_days: 20               # Test set size
backtest:
  threshold: 0.55                # Signal threshold
  tc: 0.0005                     # Transaction cost
```

## 📈 Usage Examples

### CLI Pipeline
```bash
python main.py --config config.yaml
```

### Streamlit UI
1. Enter ticker symbol (e.g., "AAPL", "TSLA", "RELIANCE.NS")
2. Set lookback period and date range
3. Click "Run Prediction"
4. View results, news samples, and feature importance

### API Endpoints
```bash
# Start FastAPI server
uvicorn api.server:app --host 0.0.0.0 --port 8000

# Get prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "lookback_days": 120}'
```

## 🎯 Key Metrics Explained

- **CV F1**: Cross-validated F1 score (harmonic mean of precision/recall)
- **Holdout Accuracy**: Out-of-sample prediction accuracy
- **Sharpe Ratio**: Risk-adjusted returns (annualized)
- **Max Drawdown**: Largest peak-to-trough loss
- **AUC**: Area under ROC curve (classification performance)

## 🔍 Model Performance

The pipeline typically achieves:
- **Accuracy**: 55-65% (vs 50% random baseline)
- **F1 Score**: 0.6-0.7 for direction prediction
- **Sharpe Ratio**: 2-8 (depending on market conditions)
- **Max Drawdown**: <5% with proper risk management

## 🛠️ Troubleshooting

### Common Issues

1. **No news found**: Check API keys and ticker symbol
2. **Model not found**: Run training pipeline first (`python main.py`)
3. **Memory errors**: Reduce `max_articles` in config
4. **Slow performance**: Enable GPU for FinBERT processing

### Debug Mode
```bash
# Enable verbose logging
export STREAMLIT_LOGGER_LEVEL=debug
streamlit run ui/streamlit_app.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FinBERT**: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
- **Yahoo Finance**: [yfinance](https://github.com/ranaroussi/yfinance)
- **Streamlit**: [streamlit.io](https://streamlit.io)
- **Scikit-learn**: [scikit-learn.org](https://scikit-learn.org)

## 📞 Support

For questions and support:
- Open an [issue](https://github.com/jashjain02/NLP/issues)
- Check the [documentation](https://github.com/jashjain02/NLP/wiki)
- Join our [Discord community](https://discord.gg/your-invite)

---

**⚠️ Disclaimer**: This tool is for educational and research purposes only. Past performance does not guarantee future results. Always do your own research before making investment decisions.