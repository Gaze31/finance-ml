# =============================================================================
# FINANCIAL SENTIMENT ANALYSIS — ALL 4 STAGES
# =============================================================================
# Run: python financial_sentiment.py
#
# Install deps first:
#   pip install numpy pandas scikit-learn torch transformers requests joblib
#
# Stages:
#   1. TF-IDF + Logistic Regression (baseline, ~30 sec)
#   2. BiLSTM with PyTorch (~10-15 min on CPU)
#   3. FinBERT fine-tuning (~45 min on CPU — use Colab GPU if possible)
#   4. Reality check on real scraped headlines
# =============================================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import re
import joblib
import requests
import xml.etree.ElementTree as ET
import warnings
warnings.filterwarnings("ignore")

from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
LABEL_NAMES = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
LABEL_LIST = ["negative", "neutral", "positive"]

print(f"Device: {DEVICE}")
if DEVICE.type == "cpu":
    print("WARNING: Stage 2 & 3 will be slow on CPU. Stage 3 -> use Google Colab.")


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def load_data():
    """
    Downloads FinancialPhraseBank (75% agreement) directly as a text file.
    No HuggingFace datasets library needed.
    Caches to financial_phrasebank.csv after first download.
    """
    print("\nLoading FinancialPhraseBank...")
    csv_path = "financial_phrasebank.csv"

    if not os.path.exists(csv_path):
        print("Downloading dataset (first run only)...")
        urls = [
            ("https://raw.githubusercontent.com/pnizam/Financial-Sentiment-Analysis"
             "/main/data/Sentences_75Agree.txt"),
            ("https://raw.githubusercontent.com/ankur-gupta-29"
             "/financial-news-sentiment/main/Sentences_75Agree.txt"),
        ]
        r = None
        for url in urls:
            try:
                r = requests.get(url, timeout=20)
                if r.status_code == 200:
                    break
            except Exception:
                continue

        if r is None or r.status_code != 200:
            raise RuntimeError(
                "Could not download dataset automatically.\n"
                "Please manually download 'Sentences_75Agree.txt' from:\n"
                "https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news\n"
                f"and place it in: {os.getcwd()}\n"
                "Then re-run the script."
            )

        rows = []
        for line in r.text.strip().split("\n"):
            line = line.strip()
            if "@" not in line:
                continue
            parts = line.rsplit("@", 1)
            if len(parts) != 2:
                continue
            text, sentiment = parts[0].strip(), parts[1].strip().lower()
            label_map = {"negative": 0, "neutral": 1, "positive": 2}
            if sentiment in label_map:
                rows.append({"text": text, "label": label_map[sentiment]})

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"Downloaded and cached {len(df)} samples -> {csv_path}")
    else:
        df = pd.read_csv(csv_path)
        print(f"Loaded from cache: {len(df)} samples ({csv_path})")

    print(df["label"].value_counts().rename({0: "negative", 1: "neutral", 2: "positive"}))
    return df


def preprocess(text, lowercase=True):
    if lowercase:
        text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================================================
# STAGE 1: TF-IDF + LOGISTIC REGRESSION
# =============================================================================

def run_stage1(df):
    section("STAGE 1: TF-IDF + Logistic Regression")

    df["clean"] = df["text"].apply(lambda x: preprocess(x, lowercase=True))

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["clean"], df["label"],
        test_size=0.2, random_state=SEED, stratify=df["label"]
    )

    # ngram_range=(1,2): captures bigrams like "not growing" (negation matters)
    # sublinear_tf=True: log(TF) prevents word-repetition from dominating
    # min_df=2: ignore words in <2 docs (noise)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), max_features=10000,
        min_df=2, sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(X_train_raw)  # fit ONLY on train
    X_test = vectorizer.transform(X_test_raw)        # transform only on test
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw = dict(zip(np.unique(y_train), weights))

    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                               class_weight=cw)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=LABEL_LIST))

    cm = pd.DataFrame(confusion_matrix(y_test, y_pred),
                      index=LABEL_LIST, columns=LABEL_LIST)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(cm)

    feature_names = np.array(vectorizer.get_feature_names_out())
    print("\nTop 8 features per class:")
    for i, label in enumerate(LABEL_LIST):
        top = np.argsort(model.coef_[i])[-8:][::-1]
        print(f"  [{label}]: {', '.join(feature_names[top])}")

    joblib.dump(model, "logreg_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("\nSaved: logreg_model.pkl, tfidf_vectorizer.pkl")

    return model, vectorizer


# =============================================================================
# STAGE 2: BiLSTM
# =============================================================================

class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}

    def build(self, texts):
        counter = Counter(w for t in texts for w in t.split())
        for word, freq in counter.items():
            if freq >= self.min_freq:
                self.word2idx[word] = len(self.word2idx)
        print(f"Vocabulary size: {len(self.word2idx)}")

    def encode(self, text, max_len=64):
        tokens = text.split()[:max_len]
        ids = [self.word2idx.get(t, 1) for t in tokens]
        ids += [0] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.word2idx)


class FinancialDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=64):
        self.X = [vocab.encode(t, max_len) for t in texts]
        self.y = labels.tolist()

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx], dtype=torch.long),
                torch.tensor(self.y[idx], dtype=torch.long))


class LSTMModel(nn.Module):
    # BiLSTM: reads left-to-right AND right-to-left
    # Final hidden state = concat(forward, backward) -> hidden_dim * 2
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128,
                 num_layers=2, num_classes=3, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(x)
        out = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(out))


def run_stage2(df):
    section("STAGE 2: BiLSTM (PyTorch)")

    df["clean"] = df["text"].apply(lambda x: preprocess(x, lowercase=True))
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["clean"], df["label"],
        test_size=0.2, random_state=SEED, stratify=df["label"]
    )

    vocab = Vocabulary(min_freq=2)
    vocab.build(X_train_raw)

    train_ds = FinancialDataset(X_train_raw, y_train, vocab)
    test_ds = FinancialDataset(X_test_raw, y_test, vocab)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    model = LSTMModel(len(vocab)).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float).to(DEVICE)
    )
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_loss = float("inf")
    for epoch in range(1, 16):
        model.train()
        t_loss, correct, total = 0, 0, 0
        for texts, labels in train_loader:
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(texts)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)

        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for texts, labels in test_loader:
                texts, labels = texts.to(DEVICE), labels.to(DEVICE)
                out = model(texts)
                v_loss += criterion(out, labels).item()
                preds = out.argmax(1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        scheduler.step(v_loss)
        print(f"Epoch {epoch:02d} | Train Loss: {t_loss/len(train_loader):.4f} "
              f"Acc: {correct/total:.3f} | Val Loss: {v_loss/len(test_loader):.4f} "
              f"Acc: {v_correct/v_total:.3f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), "best_lstm_model.pt")

    model.load_state_dict(torch.load("best_lstm_model.pt", map_location=DEVICE))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)
            preds = model(texts).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nFinal Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=LABEL_LIST))

    joblib.dump(vocab, "lstm_vocab.pkl")
    print("Saved: best_lstm_model.pt, lstm_vocab.pkl")

    return model, vocab


# =============================================================================
# STAGE 3: FinBERT FINE-TUNING
# =============================================================================

class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], max_length=self.max_len,
                             padding="max_length", truncation=True,
                             return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def run_stage3(df):
    section("STAGE 3: FinBERT Fine-Tuning")
    print("Downloading FinBERT (~400MB on first run)...")

    # DO NOT lowercase for BERT — pre-trained on cased text
    df["clean"] = df["text"].apply(lambda x: preprocess(x, lowercase=False))
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["clean"], df["label"],
        test_size=0.2, random_state=SEED, stratify=df["label"]
    )

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained(
        "ProsusAI/finbert", num_labels=3, ignore_mismatched_sizes=True
    ).to(DEVICE)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_ds = BERTDataset(X_train_raw, y_train, tokenizer)
    test_ds = BERTDataset(X_test_raw, y_test, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16)

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * 4
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    all_preds, all_labels = [], []

    for epoch in range(1, 5):
        model.train()
        t_loss, correct, total = 0, 0, 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            t_loss += out.loss.item()
            correct += (out.logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                v_loss += out.loss.item()
                preds = out.logits.argmax(1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        print(f"Epoch {epoch}/4 | Train Loss: {t_loss/len(train_loader):.4f} "
              f"Acc: {correct/total:.3f} | Val Loss: {v_loss/len(test_loader):.4f} "
              f"Acc: {v_correct/v_total:.3f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            model.save_pretrained("best_finbert_model")
            tokenizer.save_pretrained("best_finbert_model")

    print("\nFinal Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=LABEL_LIST))
    print("Saved: best_finbert_model/")

    return model, tokenizer


# =============================================================================
# STAGE 4: REALITY CHECK
# =============================================================================

FALLBACK_HEADLINES = [
    "Apple beats Q4 earnings expectations, raises dividend",
    "Tesla shares fall 8% after delivery miss",
    "Fed holds rates steady amid inflation concerns",
    "Amazon layoffs continue as cost-cutting accelerates",
    "Goldman Sachs upgrades Meta to buy with $600 target",
    "Oil prices drop on recession fears",
    "Nvidia data center revenue surges 400% year-over-year",
    "Regional bank failures spark contagion fears",
    "S&P 500 posts worst quarter since 2022",
    "Retail sales flat, consumer spending remains resilient",
    "Earnings missed but management raised full-year guidance",
    "Stock down 3% despite strong fundamentals, analysts say oversold",
]


def scrape_headlines(n=30):
    urls = [
        "https://finance.yahoo.com/news/rssindex",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US",
    ]
    headlines = []
    for url in urls:
        try:
            r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            root = ET.fromstring(r.content)
            for item in root.iter("item"):
                t = item.find("title")
                if t is not None and t.text:
                    headlines.append(t.text.strip())
        except Exception as e:
            print(f"RSS failed ({url}): {e}")
    headlines = list(dict.fromkeys(headlines))[:n]
    if not headlines:
        print("Scraping failed. Using fallback headlines.")
        headlines = FALLBACK_HEADLINES
    print(f"Headlines loaded: {len(headlines)}")
    return headlines


def run_stage4(lr_model, vectorizer, lstm_model, vocab, fb_model, fb_tokenizer):
    section("STAGE 4: Reality Check on Real Headlines")

    headlines = scrape_headlines()

    # LogReg
    X = vectorizer.transform([preprocess(h, lowercase=True) for h in headlines])
    lr_preds = lr_model.predict(X)

    # LSTM
    lstm_model.eval()
    encoded = [vocab.encode(preprocess(h, lowercase=True)) for h in headlines]
    tensor = torch.tensor(encoded, dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        lstm_preds = lstm_model(tensor).argmax(1).cpu().numpy()

    # FinBERT
    fb_model.eval()
    enc = fb_tokenizer(
        headlines, max_length=128, padding=True,
        truncation=True, return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        fb_preds = fb_model(**enc).logits.argmax(1).cpu().numpy()

    print(f"\n{'Headline':<72} {'LogReg':<11} {'LSTM':<11} {'FinBERT':<11} Status")
    print("-" * 110)
    disagreements = 0
    for i, h in enumerate(headlines):
        labels = [LABEL_NAMES[lr_preds[i]], LABEL_NAMES[lstm_preds[i]], LABEL_NAMES[fb_preds[i]]]
        disagree = len(set(labels)) > 1
        if disagree:
            disagreements += 1
        status = "DISAGREE" if disagree else "agree"
        print(f"{h[:72]:<72} {labels[0]:<11} {labels[1]:<11} {labels[2]:<11} {status}")

    print(f"\nDisagreements: {disagreements}/{len(headlines)} "
          f"({disagreements / len(headlines) * 100:.1f}%)")
    print("> 30% disagreement = not production-ready.")
    print("< 10% disagreement = might all be wrong the same way.")

    print("\n" + "=" * 60)
    print("QUESTIONS TO ANSWER NOW:")
    print("  1. Stage 1 macro F1 vs Stage 2 vs Stage 3")
    print("  2. Which class does each model fail on most?")
    print("  3. Pick 5 disagreement headlines — who is right?")
    print("  4. Find one headline where ALL 3 are wrong. Explain why.")
    print("  5. Would you trade on this model's output? Why or why not?")
    print("=" * 60)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    df = load_data()

    # Stage 1 — fast, always run
    lr_model, vectorizer = run_stage1(df)

    # Stage 2 — ~10-15 min on CPU
    lstm_model, vocab = run_stage2(df)

    # Stage 3 — ~45 min on CPU, use Colab if needed
    fb_model, fb_tokenizer = run_stage3(df)

    # Stage 4 — needs all 3 models above
    run_stage4(lr_model, vectorizer, lstm_model, vocab, fb_model, fb_tokenizer)