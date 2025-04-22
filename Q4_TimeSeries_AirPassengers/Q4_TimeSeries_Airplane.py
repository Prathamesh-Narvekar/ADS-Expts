# pip install pandas matplotlib statsmodels
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Step 1: Load the Dataset
data = pd.read_csv('AirPassengers.csv')

# Step 2: Preprocess the Data
# Convert the 'Month' column to datetime format
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)

# Ensure the data is sorted by date
data = data.sort_index()

# Step 3: Decompose the Time Series
# Decompose the time series using additive model
decomposition = seasonal_decompose(data['#Passengers'], model='additive')

# Step 4: Visualize the Components
plt.figure(figsize=(12, 8))

# Plot the original time series
plt.subplot(411)
plt.plot(data['#Passengers'], label='Original', color='blue')
plt.title('Original Time Series')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()

# Plot the trend component
plt.subplot(412)
plt.plot(decomposition.trend, label='Trend', color='orange')
plt.title('Trend Component')
plt.xlabel('Date')
plt.ylabel('Trend')
plt.legend()

# Plot the seasonal component
plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonal', color='green')
plt.title('Seasonal Component')
plt.xlabel('Date')
plt.ylabel('Seasonality')
plt.legend()

# Plot the residual component
plt.subplot(414)
plt.plot(decomposition.resid, label='Residual', color='red')
plt.title('Residual Component')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.legend()

plt.tight_layout()
plt.show()