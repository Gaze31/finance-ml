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

## Financial Sentiment Analysis — 4-Stage NLP Pipeline

A progressive NLP pipeline for financial sentiment classification, moving from a simple baseline to a fine-tuned transformer. Trained on FinancialPhraseBank (3,453 sentences, 75% agreement) and stress-tested against 30 live Yahoo Finance headlines.

---

### Pipeline Architecture

| Stage | Model | Approach |
|---|---|---|
| 1 | TF-IDF + Logistic Regression | Baseline — bigrams, class balancing |
| 2 | BiLSTM (PyTorch) | Sequential — custom vocabulary, 2-layer bidirectional |
| 3 | FinBERT (fine-tuned) | Transformer — ProsusAI/finbert, 4 epochs |
| 4 | Reality Check | Live Yahoo Finance headlines, 3-model comparison |

---

### Results on FinancialPhraseBank Test Set

| Model | Accuracy | Macro F1 | Parameters |
|---|---|---|---|
| TF-IDF + LogReg | 85% | 0.80 | ~9K features |
| BiLSTM | 80% | 0.75 | 991,355 |
| FinBERT (fine-tuned) | 97% | 0.96 | 109,484,547 |

---

### Stage 4 — Reality Check on Live Headlines

![Stage 4 Results](results/sentiment_stage4_results.png)

**Disagreements: 27/30 (90%)** — well above the not production-ready threshold.

---

### Key Findings

**1. Simple beats deep on small data**
LogReg (Macro F1: 0.80) outperformed BiLSTM (0.75) despite having no sequence modeling. With only 3,453 training samples, the BiLSTM overfits — 991K parameters is too many for this dataset size. Deep learning needs more data to win.

**2. FinBERT wins due to transfer learning, not architecture**
97% accuracy comes from pre-training on millions of financial documents before fine-tuning. It is not a fair comparison to Stage 1 and 2 — FinBERT already knew financial language. Fine-tuning on 3,453 samples just calibrated the output layer.

**3. 90% real-world disagreement reveals a critical gap**
LogReg over-predicts POSITIVE on live headlines — Yahoo Finance headlines use promotional language ("surges", "rallies", "best") even for neutral stories. The model learned these words as positive signals from training data but they appear in neutral context on real headlines. This is a domain shift problem — FinancialPhraseBank sentences and news headlines have different linguistic structures.

**4. Not production-ready — and here's why that matters**
90% disagreement between three models means none of them should be trusted for live trading signals. A production sentiment system would need: (a) fine-tuning on actual news headlines, not phrase-bank sentences, (b) confidence thresholding — only act on high-confidence predictions, (c) ensemble logic — only trade when 2 or 3 models agree.

---

### Would you trade on this output?

No. The 90% disagreement rate makes this unsuitable for live signal generation. However, FinBERT's 97% accuracy on clean labeled data shows the architecture is sound. The gap between lab performance and real-world performance is itself the finding — closing that gap requires headline-specific training data and walk-forward validation.

---

### Setup

```bash
pip install torch transformers scikit-learn pandas numpy requests joblib
python financial_sentiment.py
```

Stage 3 takes ~45 minutes on CPU. Use Google Colab with GPU for Stage 3 — reduces to ~5 minutes.

GridWorld Q-Learning
Tabular Q-learning agent trained on a custom 9×9 maze with walls, step penalties, and a goal reward. Implements epsilon-greedy exploration with multiplicative decay over 3,000 episodes. Agent successfully learns the optimal path from start to goal, confirmed by greedy rollout. Value map shows correct spatial gradient — high values near goal, decaying with distance.
Results: policy.png | value_map.png

DQN Stock Trading Agent — Pure NumPy Implementation
Double DQN trading agent built entirely in NumPy — no PyTorch or TensorFlow. Forward pass, backpropagation, Adam optimizer, and Huber loss all implemented from scratch. Trained on 1,588 days of regime-switching synthetic price data (bull/bear/sideways phases).
Results on 398-day unseen test set:

Agent return: -0.26% | Buy & Hold: -7.09% | Alpha: +6.83%
Max drawdown: -0.74% vs market drawdown of -7%
Agent learned capital preservation during bear market — stayed flat instead of losing

Training convergence: trades reduced from 800/episode to under 50 as epsilon decayed, returns improved from -2% to +1%.

ctor-Critic (A2C) Stock Trading Agent — Pure NumPy
On-policy A2C agent with separate actor and critic networks, GAE-λ advantage estimation, entropy bonus with decay, and action masking. Implemented entirely in NumPy.
Compared against DQN on identical test environment:
DQNA2CAlpha vs B&H+6.83%+6.90%Max Drawdown-0.74%~flat
Both agents learned capital preservation during bear market conditions. A2C converges faster due to on-policy updates — no replay buffer warmup needed.

RL Trading Agent Trilogy — Pure NumPy Implementations
Three reinforcement learning algorithms implemented from scratch in NumPy — no PyTorch or TensorFlow. All tested on identical regime-switching price data (80/20 train/test split, 2000 days).
AlgorithmApproachAlpha vs B&HKey FeatureDQNOff-policy, value-based+6.83%Experience replay, target networkA2COn-policy, policy gradient+6.90%GAE advantage, entropy bonusPPOOn-policy, clipped surrogate+6.43%Clip + KL early stop, K epochs
All three agents learned capital preservation during bear market conditions — staying near $10,000 while buy-and-hold lost 7.09%.
Key observations:

A2C achieved highest alpha due to faster on-policy convergence
PPO's KL early stopping correctly prevented policy collapse during the ratio spike at episode 90
DQN required warmup period before learning selective trading; A2C and PPO learned faster through on-policy updates


## Author

**Sumedha Hundekar** — Finance graduate building ML systems for quantitative finance in Python.  
Contact: velvetgazeze@gmail.com# finance-ml
