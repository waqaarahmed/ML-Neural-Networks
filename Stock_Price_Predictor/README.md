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
2. Run the script `stock_price_predictor.py`.
3. Modify the `company`, `start`, and `end` variables in the script to specify the company whose stock prices you want to predict and the desired time frame.
4. The script will train the LSTM model on the historical data and plot the actual vs. predicted stock prices for the specified company. Additionally, it will print the predicted stock price for the next day.

## Data Preparation

- The script preprocesses the data by scaling the closing prices using `MinMaxScaler` to normalize the data between 0 and 1.
- It prepares the training data by creating sequences of past closing prices (`x_train`) and their corresponding next-day closing prices (`y_train`).

## Model Architecture

- The LSTM model consists of multiple layers of LSTM units followed by dropout layers to prevent overfitting.
- The model predicts the next closing price based on the input sequence of past closing prices.

## Training

- The model is compiled using the Adam optimizer and Mean Squared Error (MSE) loss function.
- It is trained on the training data for a specified number of epochs and batch size.

## Testing

- The script tests the accuracy of the trained model by fetching recent stock price data.
- It predicts the stock prices for the next day and compares them with the actual prices.
- Finally, it plots the actual vs. predicted stock prices using `matplotlib`.


## Note

- This script provides a basic implementation of stock price prediction using LSTM neural networks. The accuracy of predictions may vary based on various factors such as market conditions, model hyperparameters, and the quality of training data.
