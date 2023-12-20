import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mlp
import tensorflow as tf
import datetime as dt
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler

#Loading Data
company = 'GOOGL'
start = dt.datetime(2012,1,1)
end = dt.datetime(2022,1,1)
data = yf.download(company, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))


#preparing data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
prediction_days = 365
x_train = []
y_train = []
for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])



x_train = np.array(x_train)
y_train = np.array(y_train)
#reshaping x_train
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#building the model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
#prediction of next closing value
model.add(Dense(units=1))
#compiling
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

#Testing accuracy of the model
t_start = dt.datetime(2022,1,1)
t_end = dt.datetime.now()
t_data = yf.download(company, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))
real_prices = t_data['Close'].values
dataset = pd.concat((data['Close'], t_data['Close']), axis=0)

model_inputs = dataset[len(dataset) - len(t_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)
#making prediction
x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted_price = model.predict(x_test)
predicted_price = scaler.inverse_transform(predicted_price)
#ploting the predicted price
plt.plot(real_prices, color="black", label=f"Actual {company} Price")
plt.plot(predicted_price, color="green", label=f"Predicted {company} Price")
plt.title(f"{company} share price")
plt.xlabel('Time')
plt.ylabel(f"{company} share price")
plt.legend()
plt.show()

#next day prediction
real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs+1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")