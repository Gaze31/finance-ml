# Machine Learning & Deep Learning for Finance

A collection of ML and deep learning models applied to financial forecasting, sentiment analysis, and algorithmic decision-making. Covers classical ML through modern deep learning architectures including LSTM, GRU, CNN, and reinforcement learning.

---

## Models Implemented

| Model | File | Application |
|---|---|---|
| LSTM | `lstm_timeseries_complete.py` | Time series price forecasting |
| GRU | `gru_ex.py` | Sequential return prediction |
| RNN | `rnnforstock_price.py` | Stock price modeling |
| CNN | `cnnforimage_finance.py` | Pattern recognition on price charts |
| Neural Network (Keras) | `build_nnkeras.py` | Classification and regression |
| Neural Network (PyTorch) | `buildnnin_pytorch.py` | Custom architecture experiments |
| Random Forest | `mltrading_strategy.py` | Trading signal generation |
| DQN | `implement_dqn.py` | Reinforcement learning trading agent |
| Q-Learning | `q_learning.py` | Tabular RL for portfolio decisions |
| FinBERT Sentiment | `finbert_sentiment.py` | Financial news NLP |
| VADER NLP | `nlpwith_vader.py` | Sentiment scoring |
| Word2Vec Finance | `word2vec_Finance.py` | Financial text embeddings |

---

## Sample Output

### Gold Price Analysis and Forecasting
![Gold Price Prediction](results/gold_price_prediction.png)

Multi-model comparison of gold price forecasting — Historical price vs Linear Regression, Moving Average, and Exponential Smoothing projections with 95% confidence interval. Bottom panel shows 60-day forward projection with uncertainty bands.

---

## Key Concepts Covered

**Time Series Forecasting**
- LSTM for long-range sequential dependencies
- GRU as a computationally efficient alternative
- ARMA/SARIMA statistical baselines for comparison
- GARCH for volatility forecasting

**Deep Learning Architectures**
- Sequence-to-sequence models
- Attention mechanism implementation
- CNN for financial pattern recognition
- Transformer architecture experiments

**Reinforcement Learning**
- Deep Q-Network (DQN) trading agent
- Q-learning for discrete portfolio decisions
- Reward shaping for financial environments

**NLP for Finance**
- FinBERT: domain-specific BERT for financial sentiment
- VADER: rule-based sentiment scoring
- Word2Vec embeddings trained on financial corpus
- Text classification for news-driven signals

---

## Setup

```bash
git clone https://github.com/Gaze31/finance-ml.git
cd finance-ml
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python lstm_timeseries_complete.py
```

---

## Requirements

```
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
tensorflow>=2.12.0
torch>=2.0.0
transformers>=4.30.0
yfinance>=0.2.0
nltk>=3.8.0
gensim>=4.3.0
```

---

## Known Limitations

- All models trained on sample/historical data — not validated for live trading
- LSTM and GRU models are prone to overfitting on small financial datasets
- FinBERT sentiment uses sample news data — real signal requires live news feed integration
- DQN agent trained in simplified environment — real market has higher complexity and transaction costs
- Deep learning models require GPU for reasonable training times on large datasets

---

## Next Steps

- [ ] Walk-forward validation for all forecasting models
- [ ] Live news feed integration for FinBERT sentiment
- [ ] Multi-asset DQN portfolio agent
- [ ] Transformer-based price forecasting (replacing LSTM)
- [ ] Combine sentiment signals with technical indicators

---

## Author

**Sumedha Hundekar** — Finance graduate building ML systems for quantitative finance in Python.  
Contact: velvetgazeze@gmail.com# finance-ml
