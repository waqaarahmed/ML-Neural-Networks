# Stock Price Predictor

This Python script utilizes Long Short-Term Memory (LSTM) neural networks to predict stock prices based on historical data. 
It fetches stock price data using the Yahoo Finance API (`yfinance`) for a specified company within a given time frame. 
The LSTM model is trained on past closing prices to predict future prices.

## Requirements

- Python 3.x
- Libraries: `matplotlib`, `numpy`, `pandas`, `tensorflow`, `datetime`, `yfinance`

## Usage

1. Install the required libraries using pip:

   ```bash
   pip install matplotlib numpy pandas tensorflow yfinance
