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

# Configuration
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Fetch Nifty 500 Data
def fetch_data(ticker="^CRSLDX"):
    try:
        print(f"Fetching data for {ticker}...")
        df = yf.download(ticker, start="2010-01-01")
        
        if df.empty:
            print(f"Data for {ticker} is empty. Trying fallback ticker ^CNX500.")
            ticker = "^CNX500"
            df = yf.download(ticker, start="2010-01-01")

        if df.empty:
             raise ValueError("Could not fetch data for Nifty 500 tickers.")
        
        # Handle MultiIndex columns if present (new yfinance update)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        print(f"Data fetched successfully. Rows: {len(df)}")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# 2. Feature Engineering
def add_technical_indicators(df):
    # Calculating technical indicators
    df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['EMA_10'] = ta.trend.ema_indicator(df['Close'], window=10)
    df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    df['BB_High'] = ta.volatility.bollinger_hband(df['Close'])
    df['BB_Low'] = ta.volatility.bollinger_lband(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])

    df.dropna(inplace=True)

    # Define Target: 1 if Close (t) > Close (t-1), else 0
    df['Target'] = np.where(df['Close'] > df['Close'].shift(1), 1, 0)
    
    return df

# Dataset Class
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Model Definition
class CNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob=0.3):
        super(CNNLSTM, self).__init__()
        
        # CNN Layers
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(dropout_prob)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(dropout_prob)
        
        # LSTM Layers
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_prob)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.permute(0, 2, 1) # (batch, features, seq_len)
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout2(x)
        
        # Prepare for LSTM: (batch, seq_len, features)
        x = x.permute(0, 2, 1) # (batch, new_seq_len, 128)
        
        lstm_out, _ = self.lstm(x)
        # Take the output of the last time step
        last_out = lstm_out[:, -1, :]
        
        out = self.fc_layers(last_out)
        return out

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=100):
    train_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs_batch, labels_batch in train_loader:
            inputs_batch = inputs_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
            
            # Calculate training accuracy
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            total_train += labels_batch.size(0)
            correct_train += (predicted == labels_batch).sum().item()
        
        avg_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(avg_loss)
        
        # Validation
        model.eval()
        val_loss_sum = 0
        correct = 0
        total = 0
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        
        with torch.no_grad():
            for inputs_batch, labels_batch in test_loader:
                inputs_batch = inputs_batch.to(device)
                labels_batch = labels_batch.to(device)
                
                outputs = model(inputs_batch)
                loss = criterion(outputs, labels_batch)
                val_loss_sum += loss.item()
                
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                
                total += labels_batch.size(0)
                correct += (predicted == labels_batch).sum().item()
                
                # Metrics
                current_tp = ((predicted == 1) & (labels_batch == 1)).sum().item()
                current_tn = ((predicted == 0) & (labels_batch == 0)).sum().item()
                current_fp = ((predicted == 1) & (labels_batch == 0)).sum().item()
                current_fn = ((predicted == 0) & (labels_batch == 1)).sum().item()
                
                tp += current_tp
                tn += current_tn
                fp += current_fp
                fn += current_fn
            
            val_acc = correct / total
            val_accuracies.append(val_acc)
            
            recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            scheduler.step(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}, R1: {recall_1:.2f}, R0: {recall_0:.2f}")

    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    return train_losses, val_accuracies

def evaluate_model(model, test_loader):
    model.eval()
    y_pred_list = []
    y_true_list = []
    y_prob_list = []

    with torch.no_grad():
        for inputs_batch, labels_batch in test_loader:
            inputs_batch = inputs_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            outputs = model(inputs_batch)
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            
            y_pred_list.extend(predicted.cpu().numpy())
            y_true_list.extend(labels_batch.cpu().numpy())
            y_prob_list.extend(probs.cpu().numpy())

    y_pred = np.array(y_pred_list)
    y_true = np.array(y_true_list)
    y_prob_list = np.array(y_prob_list)

    print(f"Test Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred))

    # Confusion Matrix (optional visualization, but printed here)
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    return y_true, y_pred, y_prob_list

def main():
    # Fetch Data
    df = fetch_data("^CRSLDX") # Nifty 500
    if df is None:
        return

    # Feature Engineering
    df = add_technical_indicators(df)

    # Select features
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                    'SMA_10', 'SMA_50', 'EMA_10', 'EMA_50', 
                    'RSI', 'MACD', 'BB_High', 'BB_Low', 'ATR']

    # Shift features and define target
    X = df[feature_cols].shift(1).iloc[1:]
    y = df['Target'].iloc[1:]
    
    # Store in a new dataframe for sequencing
    data = X.copy()
    data['Target'] = y

    print(f"Data Head:\n{data.head()}")

    # Data Preparation for CNN-LSTM
    sequence_length = 60
    features = feature_cols
    target = 'Target'

    split_idx = int(0.8 * len(data))
    train_df = data.iloc[:split_idx]
    
    # Scaling
    scaler = StandardScaler()
    scaler.fit(train_df[features])
    
    data_scaled_full = scaler.transform(data[features]) 
    target_values = data[target].values

    X_seq, y_seq = [], []

    for i in range(sequence_length - 1, len(data)):
        X_seq.append(data_scaled_full[i - sequence_length + 1 : i + 1])
        y_seq.append(target_values[i])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    train_len = split_idx - (sequence_length - 1)

    X_train = X_seq[:train_len]
    y_train = y_seq[:train_len]
    X_test = X_seq[train_len:]
    y_test = y_seq[train_len:]

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Convert to Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    # Calculate Class Weights to handle imbalance
    num_pos = np.sum(y_train)
    num_neg = len(y_train) - num_pos
    # Add epsilon to avoid division by zero
    pos_weight = num_neg / (num_pos + 1e-6)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(device)
    
    print(f"Positive samples: {num_pos}, Negative samples: {num_neg}")
    print(f"Calculated pos_weight: {pos_weight:.4f}")

    # DataLoaders
    train_dataset = TimeSeriesDataset(X_train_tensor, y_train_tensor)
    test_dataset = TimeSeriesDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Instantiate Model
    input_dim = len(features)
    hidden_dim = 64
    num_layers = 2
    output_dim = 1
    dropout_prob = 0.3

    model = CNNLSTM(input_dim, hidden_dim, num_layers, output_dim, dropout_prob).to(device)
    print(model)

    # Setup Training
    # Use pos_weight to balance the classes (predict both directions accurately)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    # Train
    train_losses, val_accuracies = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=100)

    # Evaluate
    evaluate_model(model, test_loader)
    
    # Optional: Save model
    # torch.save(model.state_dict(), 'nifty500_cnn_lstm.pth')

if __name__ == "__main__":
    main()
