"""
RNN for Stock Price Prediction
Comprehensive implementation with multiple RNN architectures
Fixed version for PyTorch compatibility
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta

# ============================================================================
# PYTORCH IMPORTS
# ============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# PART 1: DATA PREPARATION
# ============================================================================

class StockDataPreparator:
    """
    Prepare stock data for RNN training
    """
    
    def __init__(self, sequence_length=60, prediction_days=1):
        """
        Initialize data preparator
        
        Parameters:
        sequence_length: Number of past days to use for prediction
        prediction_days: Number of days ahead to predict
        """
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
        
    def fetch_data(self, symbol, start_date, end_date):
        """Fetch stock data from Yahoo Finance"""
        print(f"Fetching data for {symbol}...")
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"No data found for {symbol}")
        
        print(f"Data shape: {df.shape}")
        print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
        
        return df
    
    def create_features(self, df):
        """Create technical indicator features"""
        data = df.copy()
        
        # Price features
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data['High_Low_Ratio'] = data['High'] / data['Low']
        data['Close_Open_Ratio'] = data['Close'] / data['Open']
        
        # Moving averages
        data['SMA_5'] = data['Close'].rolling(5).mean()
        data['SMA_10'] = data['Close'].rolling(10).mean()
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # Price relative to moving averages
        data['Close_SMA5_Ratio'] = data['Close'] / data['SMA_5']
        data['Close_SMA10_Ratio'] = data['Close'] / data['SMA_10']
        data['Close_SMA20_Ratio'] = data['Close'] / data['SMA_20']
        
        # Volatility
        data['Volatility'] = data['Returns'].rolling(20).std()
        
        # Volume features
        data['Volume_Change'] = data['Volume'].pct_change()
        data['Volume_SMA'] = data['Volume'].rolling(20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        
        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
        
        # RSI
        data['RSI'] = self.calculate_rsi(data['Close'])
        
        # Drop NaN values
        data = data.dropna()
        
        return data
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_sequences(self, df, target_col='Close'):
        """
        Prepare sequences for RNN training
        
        Returns:
        X: Sequences of shape (n_samples, sequence_length, n_features)
        y: Target values of shape (n_samples, prediction_days)
        dates: Corresponding dates for each sample
        """
        # Get feature columns (all numeric columns except target)
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col != target_col]
        
        # Extract features
        feature_data = df[feature_cols].values
        
        # Scale features
        feature_data_scaled = self.feature_scaler.fit_transform(feature_data)
        
        # Extract target (closing price)
        target_data = df[target_col].values.reshape(-1, 1)
        target_scaled = self.target_scaler.fit_transform(target_data)
        
        X, y, dates = [], [], []
        
        for i in range(len(feature_data_scaled) - self.sequence_length - self.prediction_days + 1):
            # Input sequence
            X.append(feature_data_scaled[i:i + self.sequence_length])
            
            # Target (future prices)
            target_idx = i + self.sequence_length + self.prediction_days - 1
            y.append(target_scaled[target_idx])
            
            # Date for this prediction
            dates.append(df.index[target_idx])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        print(f"Created {len(X)} sequences")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y, dates
    
    def prepare_data_for_training(self, symbol, start_date, end_date, 
                                  target_col='Close', test_size=0.2):
        """Complete pipeline to prepare data for training"""
        # Fetch data
        df = self.fetch_data(symbol, start_date, end_date)
        
        # Create features
        df = self.create_features(df)
        
        # Prepare sequences
        X, y, dates = self.prepare_sequences(df, target_col)
        
        # Split into train/test
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        test_dates = dates[split_idx:]
        
        print(f"\nData split:")
        print(f"Train: {len(X_train)} sequences")
        print(f"Test: {len(X_test)} sequences")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'test_dates': test_dates,
            'target_scaler': self.target_scaler
        }

# ============================================================================
# PART 2: RNN ARCHITECTURES
# ============================================================================

class SimpleRNN(nn.Module):
    """
    Simple RNN for stock price prediction
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(SimpleRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layers
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        
        # Take the last time step
        out = out[:, -1, :]
        
        # Fully connected layers
        out = self.fc(out)
        
        return out

class LSTMModel(nn.Module):
    """
    LSTM for stock price prediction
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
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
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )
        
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last time step
        out = out[:, -1, :]
        
        # Fully connected layers
        out = self.fc(out)
        
        return out

class GRUModel(nn.Module):
    """
    GRU for stock price prediction
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate GRU
        out, _ = self.gru(x, h0)
        
        # Take the last time step
        out = out[:, -1, :]
        
        # Fully connected layers
        out = self.fc(out)
        
        return out

class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM for stock price prediction
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(BidirectionalLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers (hidden_size*2 because bidirectional)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )
        
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last time step
        out = out[:, -1, :]
        
        # Fully connected layers
        out = self.fc(out)
        
        return out

# ============================================================================
# PART 3: TRAINER CLASS (FIXED)
# ============================================================================

class RNNTrainer:
    """
    Trainer for RNN models
    """
    
    def __init__(self, model, model_name, device=device):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.train_losses = []
        self.val_losses = []
        
    def train(self, train_loader, val_loader, epochs=50, learning_rate=0.001):
        """Train the model (fixed version without verbose parameter)"""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Simple learning rate scheduler (without verbose)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        print(f"\nTraining {self.model_name}...")
        print("-" * 60)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y.squeeze())
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y.squeeze())
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            self.val_losses.append(avg_val_loss)
            
            # Update learning rate
            scheduler.step()
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"Train Loss: {avg_train_loss:.6f}, "
                      f"Val Loss: {avg_val_loss:.6f}, "
                      f"LR: {current_lr:.6f}")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        print(f"Training complete!")
        
    def predict(self, X):
        """Make predictions"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions
    
    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', alpha=0.8)
        plt.plot(self.val_losses, label='Validation Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.model_name} - Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# ============================================================================
# PART 4: EVALUATION AND VISUALIZATION
# ============================================================================

class ModelEvaluator:
    """
    Evaluate and compare RNN models
    """
    
    def __init__(self):
        self.results = {}
        
    def evaluate(self, y_true, y_pred, model_name):
        """Calculate evaluation metrics"""
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Directional accuracy (up/down prediction)
        if len(y_true) > 1:
            true_direction = np.diff(y_true.flatten()) > 0
            pred_direction = np.diff(y_pred.flatten()) > 0
            directional_acc = np.mean(true_direction == pred_direction) * 100
        else:
            directional_acc = 0
        
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Directional Accuracy %': directional_acc
        }
        
        self.results[model_name] = metrics
        
        return metrics
    
    def print_results(self):
        """Print all results"""
        print("\n" + "="*70)
        print("MODEL COMPARISON RESULTS")
        print("="*70)
        
        results_df = pd.DataFrame(self.results).T
        print(results_df.round(4))
        
        return results_df
    
    def plot_predictions(self, y_true, y_pred_dict, dates, title="Stock Price Prediction"):
        """Plot predictions from multiple models"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Time series comparison
        ax1 = axes[0]
        ax1.plot(dates, y_true, label='Actual', color='black', linewidth=2)
        
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        for i, (name, y_pred) in enumerate(y_pred_dict.items()):
            ax1.plot(dates, y_pred, label=name, color=colors[i % len(colors)], 
                    linewidth=1.5, alpha=0.7, linestyle='--')
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Stock Price')
        ax1.set_title(title)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot actual vs predicted
        ax2 = axes[1]
        
        for i, (name, y_pred) in enumerate(y_pred_dict.items()):
            ax2.scatter(y_true, y_pred, label=name, alpha=0.5, s=30)
        
        # Perfect prediction line
        min_val = min(y_true.min(), min(y_pred.min() for y_pred in y_pred_dict.values()))
        max_val = max(y_true.max(), max(y_pred.max() for y_pred in y_pred_dict.values()))
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
        
        ax2.set_xlabel('Actual Price')
        ax2.set_ylabel('Predicted Price')
        ax2.set_title('Actual vs Predicted')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_error_distribution(self, y_true, y_pred_dict):
        """Plot error distributions"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Error distribution
        ax1 = axes[0]
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        for i, (name, y_pred) in enumerate(y_pred_dict.items()):
            errors = (y_pred - y_true).flatten()
            ax1.hist(errors, bins=30, alpha=0.5, label=name, color=colors[i % len(colors)])
        
        ax1.set_xlabel('Prediction Error')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Error Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Box plot of errors
        ax2 = axes[1]
        error_data = []
        labels = []
        
        for name, y_pred in y_pred_dict.items():
            errors = (y_pred - y_true).flatten()
            error_data.append(errors)
            labels.append(name)
        
        ax2.boxplot(error_data, labels=labels)
        ax2.set_ylabel('Prediction Error')
        ax2.set_title('Error Box Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# ============================================================================
# PART 5: MAIN EXECUTION
# ============================================================================

def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=32):
    """Create PyTorch data loaders"""
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def run_stock_prediction_demo():
    """Run complete stock prediction demonstration"""
    
    print("="*80)
    print("RNN FOR STOCK PRICE PREDICTION")
    print("="*80)
    
    # Parameters
    symbol = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    sequence_length = 60
    prediction_days = 1
    
    print(f"\nAnalyzing {symbol} from {start_date} to {end_date}")
    print(f"Sequence length: {sequence_length}, Prediction days: {prediction_days}")
    
    # Prepare data
    preparator = StockDataPreparator(sequence_length=sequence_length, 
                                     prediction_days=prediction_days)
    
    data_dict = preparator.prepare_data_for_training(symbol, start_date, end_date)
    
    X_train, y_train = data_dict['X_train'], data_dict['y_train']
    X_test, y_test = data_dict['X_test'], data_dict['y_test']
    test_dates = data_dict['test_dates']
    target_scaler = data_dict['target_scaler']
    
    # Split validation from training
    val_size = int(0.2 * len(X_train))
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]
    
    print(f"\nFinal split:")
    print(f"Train: {len(X_train)} sequences")
    print(f"Validation: {len(X_val)} sequences")
    print(f"Test: {len(X_test)} sequences")
    print(f"Input features: {X_train.shape[2]}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val, batch_size=32)
    
    # Dictionary to store models and predictions
    models = {}
    predictions = {}
    trainers = {}
    
    # Define models to train (using smaller hidden size for faster training)
    model_configs = [
        (SimpleRNN, "Simple RNN", 32, 2),
        (LSTMModel, "LSTM", 32, 2),
        (GRUModel, "GRU", 32, 2),
        (BidirectionalLSTM, "Bi-LSTM", 32, 2),
    ]
    
    # Train each model
    for model_class, model_name, hidden_size, num_layers in model_configs:
        print(f"\n{'-'*60}")
        print(f"Training {model_name}")
        print(f"{'-'*60}")
        
        # Initialize model
        model = model_class(
            input_size=X_train.shape[2],
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=1,
            dropout=0.2
        )
        
        # Print model size
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        # Train
        trainer = RNNTrainer(model, model_name)
        trainer.train(train_loader, val_loader, epochs=30, learning_rate=0.001)  # Fewer epochs for faster demo
        
        # Store
        models[model_name] = model
        trainers[model_name] = trainer
        
        # Make predictions
        y_pred_scaled = trainer.predict(X_test)
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
        predictions[model_name] = y_pred.flatten()
        
        # Plot training history
        trainer.plot_training_history()
    
    # Inverse transform actual values
    y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Evaluate models
    evaluator = ModelEvaluator()
    
    print("\n" + "="*70)
    print("EVALUATING MODELS")
    print("="*70)
    
    for name, y_pred in predictions.items():
        metrics = evaluator.evaluate(y_test_actual, y_pred, name)
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Print comparison
    results_df = evaluator.print_results()
    
    # Plot predictions
    evaluator.plot_predictions(y_test_actual, predictions, test_dates, 
                               f"{symbol} Stock Price Prediction")
    
    # Plot error distributions
    evaluator.plot_error_distribution(y_test_actual, predictions)
    
    # Find best model
    best_model = results_df['RMSE'].idxmin()
    print(f"\nBest model based on RMSE: {best_model}")
    
    return models, trainers, predictions, evaluator

# ============================================================================
# PART 6: SIMPLE PREDICTION EXAMPLE
# ============================================================================

def simple_prediction_example():
    """Simple example of using a trained model"""
    
    print("\n" + "="*80)
    print("SIMPLE PREDICTION EXAMPLE")
    print("="*80)
    
    # Create a simple model for demonstration
    model = LSTMModel(input_size=10, hidden_size=32, num_layers=2, output_size=1)
    
    # Create random data for demonstration
    X_demo = torch.randn(5, 60, 10)  # 5 samples, 60 time steps, 10 features
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        predictions = model(X_demo)
    
    print(f"Input shape: {X_demo.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:3].numpy().flatten()}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        # Run the main demonstration
        models, trainers, predictions, evaluator = run_stock_prediction_demo()
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETE")
        print("="*80)
        
        # Show simple example
        simple_prediction_example()
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc() 