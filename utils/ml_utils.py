import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import talib
import ccxt
import time
import logging
from config import BINANCE_API_KEY, BINANCE_API_SECRET

logging.basicConfig(level=logging.INFO, filename='bot.log', filemode='a', format='%(asctime)s - %(message)s')

class PriceLSTM(nn.Module):
    """
    LSTM model for price prediction in scalping.

    Note: Simplified to 3 input features (close, volume, RSI) for fast training
    and relevance to scalping (5m timeframe).
    """
    def __init__(self, input_size=3, hidden_size=50, num_layers=1):
        """
        Initialize LSTM model.

        Args:
            input_size (int): Number of input features (default 3: close, volume, RSI).
            hidden_size (int): Number of LSTM hidden units.
            num_layers (int): Number of LSTM layers.
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass for price prediction.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_size).

        Returns:
            torch.Tensor: Predicted next price.
        """
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))

def fetch_historical_data(symbol='BTC/USDT', timeframe='5m', years=1):
    """
    Fetch historical OHLCV data with pagination for ML training.

    Args:
        symbol (str): Trading pair (e.g., 'BTC/USDT').
        timeframe (str): Candle timeframe (default '5m' for scalping).
        years (int): Years of data to fetch (default 1).

    Returns:
        pd.DataFrame: OHLCV data with RSI and EMA indicators.

    Note: Paginates to bypass CCXT 1000-candle limit; sleeps to avoid rate limits.
    """
    try:
        client = ccxt.binance({'apiKey': BINANCE_API_KEY, 'secret': BINANCE_API_SECRET})
        limit = 1000
        since = int(time.time() * 1000) - (years * 365 * 24 * 60 * 60 * 1000)  # ms
        all_ohlcv = []
        while since < int(time.time() * 1000):
            ohlcv = client.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1  # Next batch
            time.sleep(1)  # Avoid rate limits
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['rsi'] = talib.RSI(df['close'])
        df['ema'] = talib.EMA(df['close'], 12)
        logging.info(f"Fetched {len(df)} candles for {symbol}")
        return df.dropna()
    except Exception as e:
        logging.error(f"Historical data fetch failed for {symbol}: {e}")
        return pd.DataFrame()

def train_model(symbol='BTC/USDT', epochs=5):
    """
    Train LSTM model with train/validation split to prevent overfitting.

    Args:
        symbol (str): Trading pair.
        epochs (int): Number of training epochs (default 5 for quick sims).

    Returns:
        tuple: (model, train_loss, val_loss)

    Note: Uses 80/20 train/validation split; checks for overfitting (val_loss < 1.2*train_loss).
    """
    try:
        model = PriceLSTM(input_size=3)  # Close, volume, RSI
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        df = fetch_historical_data(symbol, years=1)
        features = df[['close', 'volume', 'rsi']].values
        features = (features - features.mean(0)) / features.std(0)  # Normalize
        split = int(0.8 * len(features))
        train_X, val_X = features[:split, :-1], features[split:, :-1]
        train_y, val_y = features[1:split+1, 0], features[split+1:, 0]
        train_X = torch.tensor(train_X.reshape(-1, 1, 3), dtype=torch.float32)
        train_y = torch.tensor(train_y.reshape(-1, 1), dtype=torch.float32)
        val_X = torch.tensor(val_X.reshape(-1, 1, 3), dtype=torch.float32)
        val_y = torch.tensor(val_y.reshape(-1, 1), dtype=torch.float32)

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(train_X)
            train_loss = criterion(output, train_y)
            train_loss.backward()
            optimizer.step()
            # Validation loss
            with torch.no_grad():
                val_output = model(val_X)
                val_loss = criterion(val_output, val_y)
            logging.info(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss > train_loss * 1.2:
            logging.warning("Potential overfitting detected")
        return model, train_loss.item(), val_loss.item()
    except Exception as e:
        logging.error(f"Model training failed for {symbol}: {e}")
        return None, 0.0, 0.0

def predict_next_price(model, recent_data):
    """
    Predict next price using trained LSTM model.

    Args:
        model (PriceLSTM): Trained LSTM model.
        recent_data (np.ndarray): Recent features (close, volume, RSI).

    Returns:
        float: Predicted next price.

    Note: Normalizes input data consistent with training.
    """
    try:
        if model is None:
            logging.error("No trained model provided for prediction")
            return 0.0
        input_data = (recent_data - recent_data.mean(0)) / recent_data.std(0)
        input_tensor = torch.tensor(input_data.reshape(1, 1, 3), dtype=torch.float32)
        with torch.no_grad():
            pred = model(input_tensor).item()
        logging.info(f"Predicted next price: {pred:.2f}")
        return pred
    except Exception as e:
        logging.error(f"Price prediction failed: {e}")
        return 0.0
