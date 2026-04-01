import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Fetch Nifty 100 Data
def fetch_data(ticker="^CNX100"):
    print(f"Fetching data for {ticker}...")
    try:
        df = yf.download(ticker, start="2010-01-01")
        if df.empty:
            print("Data empty, trying fallback ^NSEI")
            df = yf.download("^NSEI", start="2010-01-01")
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# 2. Feature Engineering
def add_technical_indicators(df):
    df = df.copy()
    # Trend Indicators
    df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['EMA_10'] = ta.trend.ema_indicator(df['Close'], window=10)
    df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
    
    # Momentum Indicators
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    
    # Volatility Indicators
    df['BB_High'] = ta.volatility.bollinger_hband(df['Close'])
    df['BB_Low'] = ta.volatility.bollinger_lband(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    
    df.dropna(inplace=True)
    return df

# 3. Prepare Data for Models
def prepare_data(df, sequence_length=60):
    # Target: 1 if Close > Prev Close
    df['Target'] = np.where(df['Close'] > df['Close'].shift(1), 1, 0)
    
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                    'SMA_10', 'SMA_50', 'EMA_10', 'EMA_50', 
                    'RSI', 'MACD', 'BB_High', 'BB_Low', 'ATR']
    
    # Shift features to align X[t] with y[t+1] (Predicting tomorrow's direction using today's data)
    # The previous code used: X = df[features].shift(1), y = df['Target']
    # This means X[t] contains data from t-1, and y[t] is direction at t.
    X = df[feature_cols].shift(1).iloc[1:]
    y = df['Target'].iloc[1:]
    
    # Split Train/Test
    split_idx = int(0.8 * len(X))
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create Sequences
    X_seq, y_seq = [], []
    for i in range(sequence_length - 1, len(X_scaled)):
        X_seq.append(X_scaled[i-sequence_length+1 : i+1])
        y_seq.append(y.iloc[i])
        
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # Split into train/test
    train_len = int(0.8 * len(X_seq))
    
    X_train = torch.tensor(X_seq[:train_len], dtype=torch.float32).to(device)
    y_train = torch.tensor(y_seq[:train_len], dtype=torch.float32).unsqueeze(1).to(device)
    X_test = torch.tensor(X_seq[train_len:], dtype=torch.float32).to(device)
    y_test = torch.tensor(y_seq[train_len:], dtype=torch.float32).unsqueeze(1).to(device)
    
    return X_train, y_train, X_test, y_test, len(feature_cols)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# 4. Define Models
class CNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Calculate flatten size
        # Input: (batch, input_dim, seq_len) -> (batch, 64, seq_len/2) -> (batch, 128, seq_len/4)
        # We will determine size dynamically in forward or assume fixed seq len?
        # Let's use AdaptiveAvgPool to handle variable lengths or simplify
        self.global_pool = nn.AdaptiveAvgPool1d(1) 
        
        self.fc = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1) # Flatten
        
        out = self.fc(x)
        return out

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.3):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # Take last time step output. For bidirectional, it's concatenation of forward and backward states
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.3):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

# 5. Training Function
def train_model(model, train_loader, test_loader, num_epochs=50, lr=0.001):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    train_losses = []
    val_accuracies = []
    best_acc = 0.0
    
    print(f"Training {model.__class__.__name__}...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        val_acc = correct / total
        val_accuracies.append(val_acc)
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
            
    return train_losses, val_accuracies, best_acc

# Main Execution
def main():
    df = fetch_data()
    if df is None: return
    
    df = add_technical_indicators(df)
    print(f"Data prepared: {df.shape}")
    
    X_train, y_train, X_test, y_test, input_dim = prepare_data(df)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model Config
    hidden_dim = 64
    num_layers = 2
    output_dim = 1
    
    models = {
        'CNN': CNNModel(input_dim, hidden_dim, output_dim).to(device),
        'Bi-LSTM': BiLSTMModel(input_dim, hidden_dim, num_layers, output_dim).to(device),
        'GRU': GRUModel(input_dim, hidden_dim, num_layers, output_dim).to(device)
    }
    
    results = {}
    
    plt.figure(figsize=(15, 5))
    
    for i, (name, model) in enumerate(models.items()):
        losses, accuracies, best_acc = train_model(model, train_loader, test_loader)
        results[name] = best_acc
        print(f"{name} Best Accuracy: {best_acc:.4f}")
        
        # Plot
        plt.subplot(1, 3, i+1)
        plt.plot(losses, label='Train Loss')
        plt.plot(accuracies, label='Val Accuracy')
        plt.title(f'{name} (Best Acc: {best_acc:.4f})')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()
    
    print("\nFinal Results:")
    for name, acc in results.items():
        print(f"{name}: {acc:.4f}")

    # Detailed Evaluation for best model (Optional) but requested to evaluate performance
    # We'll just print classification report for all
    print("\nDetailed Classification Reports:")
    for name, model in models.items():
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(y_batch.cpu().numpy())
        
        print(f"--- {name} ---")
        print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    main()
