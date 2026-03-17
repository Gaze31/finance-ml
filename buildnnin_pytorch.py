"""
Neural Network Trading System with PyTorch
Fixed version with proper sequence alignment
"""

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# DATA PREPARATION
# ============================================================================

class TradingDataLoader:
    """
    Prepare and load data for neural network training
    """
    
    def __init__(self, sequence_length=60, prediction_horizon=5):
        """
        Initialize data loader
        
        Parameters:
        sequence_length: Number of past days to use for prediction
        prediction_horizon: How many days ahead to predict
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def fetch_data(self, symbols, start_date, end_date):
        """
        Fetch data for multiple symbols
        """
        print(f"Fetching data for {symbols}...")
        data = yf.download(symbols, start=start_date, end=end_date, progress=False)
        
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            if 'Adj Close' in data.columns.get_level_values(0):
                data = data['Adj Close']
            elif 'Close' in data.columns.get_level_values(0):
                data = data['Close']
        
        # Ensure we have a DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame()
            if len(symbols) == 1:
                data.columns = symbols
        
        return data
    
    def create_features(self, df):
        """
        Create technical indicator features
        """
        features = pd.DataFrame(index=df.index)
        feature_names = []
        
        for col in df.columns:
            price = df[col]
            
            # Price-based features
            feat_name = f'{col}_returns_1'
            features[feat_name] = price.pct_change(1)
            feature_names.append(feat_name)
            
            feat_name = f'{col}_returns_5'
            features[feat_name] = price.pct_change(5)
            feature_names.append(feat_name)
            
            feat_name = f'{col}_returns_20'
            features[feat_name] = price.pct_change(20)
            feature_names.append(feat_name)
            
            # Moving averages
            feat_name = f'{col}_sma_20'
            features[feat_name] = price.rolling(20).mean() / price - 1
            feature_names.append(feat_name)
            
            feat_name = f'{col}_sma_50'
            features[feat_name] = price.rolling(50).mean() / price - 1
            feature_names.append(feat_name)
            
            # Volatility
            feat_name = f'{col}_volatility'
            features[feat_name] = price.pct_change().rolling(20).std()
            feature_names.append(feat_name)
            
            # RSI
            feat_name = f'{col}_rsi'
            features[feat_name] = self.calculate_rsi(price)
            feature_names.append(feat_name)
            
            # MACD
            macd, signal = self.calculate_macd(price)
            feat_name = f'{col}_macd'
            features[feat_name] = macd
            feature_names.append(feat_name)
            
            feat_name = f'{col}_macd_signal'
            features[feat_name] = signal
            feature_names.append(feat_name)
            
            # Price position
            rolling_high = price.rolling(20).max()
            rolling_low = price.rolling(20).min()
            feat_name = f'{col}_price_position'
            features[feat_name] = (price - rolling_low) / (rolling_high - rolling_low)
            feature_names.append(feat_name)
        
        # Store feature columns for later use
        self.feature_columns = feature_names
        
        # Drop NaN values
        features = features.dropna()
        
        return features
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
    
    def prepare_sequences(self, features, target_col):
        """
        Create sequences for LSTM/CNN models
        
        Returns:
        X: Sequences of shape (n_samples, sequence_length, n_features)
        y: Targets of shape (n_samples,)
        valid_dates: Index dates corresponding to each sequence
        """
        # Get feature columns for this target
        target_prefix = target_col.split('_')[0]
        feature_cols = [col for col in features.columns if col.startswith(target_prefix)]
        
        if not feature_cols:
            # If no specific columns found, use all columns
            feature_cols = features.columns.tolist()
        
        feature_data = features[feature_cols].values
        
        # Scale features
        feature_data_scaled = self.scaler.fit_transform(feature_data)
        
        # Create target: future returns
        future_returns = features[f'{target_col}_returns_1'].shift(-self.prediction_horizon).values
        
        X, y, valid_dates = [], [], []
        
        # Create sequences
        for i in range(len(feature_data_scaled) - self.sequence_length - self.prediction_horizon + 1):
            X.append(feature_data_scaled[i:i + self.sequence_length])
            
            # Target is the future return at the end of prediction horizon
            target_idx = i + self.sequence_length + self.prediction_horizon - 1
            target_value = future_returns[target_idx]
            
            if not np.isnan(target_value):
                y.append(target_value)
                valid_dates.append(features.index[target_idx])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        # Trim X and y to same length (in case of NaN values)
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]
        valid_dates = valid_dates[:min_len]
        
        print(f"Created {len(X)} sequences")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y, valid_dates

# ============================================================================
# NEURAL NETWORK ARCHITECTURES
# ============================================================================

class LSTMModel(nn.Module):
    """
    LSTM-based neural network for sequence prediction
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Output: prediction
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(32)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last time step
        out = out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        
        return out.squeeze()

class TransformerModel(nn.Module):
    """
    Transformer-based neural network for sequence prediction
    """
    
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.fc1 = nn.Linear(d_model, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Output layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x.squeeze()

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models
    """
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class CNNModel(nn.Module):
    """
    CNN-based neural network for sequence prediction
    """
    
    def __init__(self, input_size, sequence_length, num_filters=64, kernel_size=3):
        super(CNNModel, self).__init__()
        
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size, padding=1)
        self.conv2 = nn.Conv1d(num_filters, num_filters*2, kernel_size, padding=1)
        self.conv3 = nn.Conv1d(num_filters*2, num_filters*4, kernel_size, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool1d(2)
        
        # Calculate size after convolutions and pooling
        conv_output_size = self._get_conv_output_size(sequence_length)
        
        # Fully connected layers
        self.fc1 = nn.Linear(num_filters*4 * conv_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)
        
    def _get_conv_output_size(self, sequence_length):
        """Calculate the size after convolutional layers"""
        size = sequence_length
        for _ in range(3):  # 3 pooling layers
            size = size // 2
        return max(size, 1)
    
    def forward(self, x):
        # Reshape for Conv1d: (batch, channels, sequence_length)
        x = x.transpose(1, 2)
        
        # Convolutional layers
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        
        return x.squeeze()

# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

class NeuralTradingTrainer:
    """
    Trainer class for neural network models
    """
    
    def __init__(self, model, device='cpu'):  # Force CPU since CUDA not available
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = None
        self.scheduler = None
        self.train_losses = []
        self.val_losses = []
        
    def prepare_data(self, X_train, y_train, X_val, y_val, batch_size=32):
        """Prepare data loaders"""
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
    def train(self, epochs=100, learning_rate=0.001, patience=10):
        """Train the model"""
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in self.train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in self.val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(self.val_loader)
            self.val_losses.append(avg_val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return self.train_losses, self.val_losses
    
    def predict(self, X):
        """Make predictions"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions
    
    def plot_training_history(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', alpha=0.8)
        plt.plot(self.val_losses, label='Validation Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# ============================================================================
# TRADING STRATEGY WITH NEURAL NETWORKS
# ============================================================================

class NeuralTradingStrategy:
    """
    Trading strategy using neural network predictions
    """
    
    def __init__(self, model_type='lstm', sequence_length=60, prediction_horizon=5):
        """
        Initialize neural trading strategy
        
        Parameters:
        model_type: 'lstm', 'transformer', or 'cnn'
        sequence_length: Number of past days for prediction
        prediction_horizon: Days ahead to predict
        """
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.trainer = None
        self.data_loader = TradingDataLoader(sequence_length, prediction_horizon)
        self.test_dates = None
        
    def build_model(self, input_size):
        """Build neural network based on specified type"""
        if self.model_type == 'lstm':
            self.model = LSTMModel(input_size=input_size)
        elif self.model_type == 'transformer':
            self.model = TransformerModel(input_size=input_size)
        elif self.model_type == 'cnn':
            self.model = CNNModel(input_size=input_size, sequence_length=self.sequence_length)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.trainer = NeuralTradingTrainer(self.model)
        return self.model
    
    def prepare_training_data(self, df, target_symbol):
        """Prepare data for training"""
        # Create features
        features = self.data_loader.create_features(df)
        print(f"Features shape: {features.shape}")
        print(f"Feature columns: {features.columns.tolist()}")
        
        # Create sequences
        X, y, valid_dates = self.data_loader.prepare_sequences(features, target_symbol)
        
        # Split into train/val/test
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        # Store test dates for later use
        self.test_dates = valid_dates[train_size+val_size:]
        
        print(f"\nData split:")
        print(f"Train: {len(X_train)} samples")
        print(f"Validation: {len(X_val)} samples")
        print(f"Test: {len(X_test)} samples")
        print(f"Test dates range: {self.test_dates[0]} to {self.test_dates[-1]}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the neural network"""
        self.trainer.prepare_data(X_train, y_train, X_val, y_val, batch_size)
        train_losses, val_losses = self.trainer.train(epochs=epochs)
        return train_losses, val_losses
    
    def generate_signals(self, X_test, threshold=0.01):
        """Generate trading signals from predictions"""
        predictions = self.trainer.predict(X_test)
        
        # Ensure predictions and test_dates have the same length
        min_len = min(len(predictions), len(self.test_dates))
        predictions = predictions[:min_len]
        test_dates = self.test_dates[:min_len]
        
        signals = pd.DataFrame(index=test_dates)
        signals['prediction'] = predictions
        signals['signal'] = 0
        
        # Generate signals based on predictions
        signals.loc[signals['prediction'] > threshold, 'signal'] = 1  # Buy
        signals.loc[signals['prediction'] < -threshold, 'signal'] = -1  # Sell
        
        # Add confidence based on prediction magnitude
        signals['confidence'] = abs(signals['prediction']) * 100
        
        print(f"Generated {len(signals)} signals")
        print(f"Buy signals: {(signals['signal'] == 1).sum()}")
        print(f"Sell signals: {(signals['signal'] == -1).sum()}")
        
        return signals
    
    def backtest(self, signals, prices, initial_capital=100000, transaction_cost=0.001):
        """Backtest the strategy"""
        backtest = signals.copy()
        
        # Align prices with signals
        common_dates = backtest.index.intersection(prices.index)
        backtest = backtest.loc[common_dates]
        backtest['price'] = prices.loc[common_dates]
        
        # Calculate returns
        backtest['returns'] = backtest['price'].pct_change()
        backtest['strategy_returns'] = backtest['signal'].shift(1) * backtest['returns']
        
        # Apply transaction costs
        backtest['position_changed'] = backtest['signal'].diff().fillna(0) != 0
        backtest.loc[backtest['position_changed'], 'strategy_returns'] -= transaction_cost
        
        # Calculate equity curves
        backtest['cumulative_returns'] = (1 + backtest['strategy_returns']).cumprod()
        backtest['buy_hold_returns'] = (1 + backtest['returns']).cumprod()
        backtest['strategy_value'] = initial_capital * backtest['cumulative_returns']
        
        # Calculate metrics
        metrics = self.calculate_metrics(backtest['strategy_returns'].dropna(), 
                                        backtest['returns'].dropna())
        
        return backtest, metrics
    
    def calculate_metrics(self, strategy_returns, benchmark_returns):
        """Calculate performance metrics"""
        metrics = {}
        
        # Return metrics
        metrics['Total Return'] = (1 + strategy_returns).prod() - 1
        metrics['Benchmark Return'] = (1 + benchmark_returns).prod() - 1
        
        # Annualized metrics
        years = len(strategy_returns) / 252
        metrics['Annualized Return'] = (1 + metrics['Total Return']) ** (1/years) - 1 if years > 0 else 0
        metrics['Annualized Volatility'] = strategy_returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        metrics['Sharpe Ratio'] = metrics['Annualized Return'] / metrics['Annualized Volatility'] \
                                  if metrics['Annualized Volatility'] != 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics['Max Drawdown'] = drawdown.min()
        
        # Win rate
        winning_trades = strategy_returns[strategy_returns > 0].count()
        total_trades = strategy_returns[strategy_returns != 0].count()
        metrics['Win Rate'] = winning_trades / total_trades if total_trades > 0 else 0
        
        # Number of trades
        metrics['Number of Trades'] = total_trades
        
        return metrics
    
    def plot_results(self, backtest, model_name):
        """Plot backtest results"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name} Neural Trading Strategy - {self.model_type.upper()}', fontsize=16)
        
        # Price and signals
        ax1 = axes[0, 0]
        ax1.plot(backtest.index, backtest['price'], color='black', alpha=0.7, linewidth=1)
        
        buy_signals = backtest[backtest['signal'] == 1]
        sell_signals = backtest[backtest['signal'] == -1]
        
        if not buy_signals.empty:
            ax1.scatter(buy_signals.index, buy_signals['price'], 
                       marker='^', color='green', s=30, label='Buy', alpha=0.7)
        if not sell_signals.empty:
            ax1.scatter(sell_signals.index, sell_signals['price'], 
                       marker='v', color='red', s=30, label='Sell', alpha=0.7)
        
        ax1.set_title('Price with Trading Signals')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Predictions
        ax2 = axes[0, 1]
        ax2.plot(backtest.index, backtest['prediction'], color='blue', linewidth=1)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.axhline(y=0.01, color='green', linestyle='--', alpha=0.5, label='Buy Threshold')
        ax2.axhline(y=-0.01, color='red', linestyle='--', alpha=0.5, label='Sell Threshold')
        ax2.set_title('Model Predictions')
        ax2.set_ylabel('Predicted Return')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Equity curve
        ax3 = axes[1, 0]
        ax3.plot(backtest.index, backtest['cumulative_returns'], 
                label='Strategy', color='blue', linewidth=2)
        ax3.plot(backtest.index, backtest['buy_hold_returns'], 
                label='Buy & Hold', color='gray', linewidth=1, alpha=0.7, linestyle='--')
        ax3.set_title('Equity Curve')
        ax3.set_ylabel('Cumulative Returns')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Drawdown
        ax4 = axes[1, 1]
        cumulative = backtest['cumulative_returns']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        ax4.fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3)
        ax4.plot(drawdown.index, drawdown, color='red', alpha=0.7)
        ax4.set_title('Drawdown')
        ax4.set_ylabel('Drawdown (%)')
        ax4.grid(True, alpha=0.3)
        
        # Confidence distribution
        ax5 = axes[2, 0]
        ax5.hist(backtest['confidence'].dropna(), bins=30, color='purple', alpha=0.7)
        ax5.set_title('Signal Confidence Distribution')
        ax5.set_xlabel('Confidence')
        ax5.set_ylabel('Frequency')
        ax5.grid(True, alpha=0.3)
        
        # Returns distribution
        ax6 = axes[2, 1]
        ax6.hist(backtest['strategy_returns'].dropna() * 100, bins=30, color='blue', alpha=0.7)
        ax6.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax6.set_title('Strategy Returns Distribution')
        ax6.set_xlabel('Daily Return (%)')
        ax6.set_ylabel('Frequency')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print metrics
        print("\n" + "="*50)
        print("PERFORMANCE METRICS")
        print("="*50)
        metrics = self.calculate_metrics(backtest['strategy_returns'].dropna(), 
                                        backtest['returns'].dropna())
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'Ratio' in key or 'Rate' in key:
                    print(f"{key:<25}: {value:.4f}")
                else:
                    print(f"{key:<25}: {value:.2%}")
            else:
                print(f"{key:<25}: {value}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_neural_trading_demo():
    """
    Run a complete demonstration of neural trading strategies
    """
    print("="*70)
    print("NEURAL NETWORK TRADING SYSTEM WITH PYTORCH")
    print("="*70)
    
    # Parameters
    symbol = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    
    print(f"\nAnalyzing {symbol} from {start_date} to {end_date}")
    
    # Initialize strategy
    strategy = NeuralTradingStrategy(model_type='lstm', sequence_length=60, prediction_horizon=5)
    
    # Fetch data
    data_loader = TradingDataLoader()
    df = data_loader.fetch_data([symbol], start_date, end_date)
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Prepare data
    print("\nPreparing data for training...")
    train_data, val_data, test_data = strategy.prepare_training_data(df, symbol)
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    # Build model
    print(f"\nBuilding {strategy.model_type.upper()} model...")
    strategy.build_model(input_size=X_train.shape[2])
    print(f"Model architecture:\n{strategy.model}")
    
    # Train model
    print("\nTraining model...")
    train_losses, val_losses = strategy.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
    
    # Plot training history
    strategy.trainer.plot_training_history()
    
    # Generate signals
    print("\nGenerating trading signals...")
    signals = strategy.generate_signals(X_test, threshold=0.01)
    
    # Backtest
    test_prices = df.loc[signals.index, symbol]
    backtest, metrics = strategy.backtest(signals, test_prices)
    
    # Plot results
    strategy.plot_results(backtest, strategy.model_type.upper())
    
    return strategy, backtest, metrics

def main():
    """
    Main function to run neural trading system
    """
    print("="*80)
    print("NEURAL NETWORK TRADING SYSTEM WITH PYTORCH")
    print("="*80)
    
    # Run demonstration
    strategy, backtest, metrics = run_neural_trading_demo()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return strategy

if __name__ == "__main__":
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Run main
    strategy = main()