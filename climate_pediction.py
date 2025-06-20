from google.colab import files
files.upload()

Upload your CSV file (GlobalLandTemperaturesByCountry.csv) to Colab using the above code

# 📌 Step 1: Install required libraries
!pip install prophet --quiet

# 📌 Step 2: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# 📌 Step 3: Load the dataset
df = pd.read_csv('/content/GlobalLandTemperaturesByCountry.csv')

# 📌 Step 4: Preprocessing
df['dt'] = pd.to_datetime(df['dt'])
df = df.dropna(subset=['AverageTemperature'])
df_india = df[df['Country'] == 'India']  # You can change the country here
df_india['Year'] = df_india['dt'].dt.year
df_india['Month'] = df_india['dt'].dt.month
annual_temp = df_india.groupby('Year')['AverageTemperature'].mean().reset_index()

# 📌 Step 5: Data Visualization
plt.figure(figsize=(12,6))
sns.lineplot(data=annual_temp, x='Year', y='AverageTemperature', marker='o')
plt.title('Average Annual Temperature in India')
plt.xlabel('Year')
plt.ylabel('Average Temperature (°C)')
plt.grid(True)
plt.show()

# 📌 Step 6: Machine Learning - Linear Regression
X = annual_temp[['Year']]
y = annual_temp['AverageTemperature']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("✅ Mean Squared Error:", mse)

# 📌 Step 7: Prediction Visualization
plt.figure(figsize=(12,6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Year')
plt.ylabel('Average Temperature')
plt.title('Predicted vs Actual Temperatures')
plt.legend()
plt.grid(True)
plt.show()

# 📌 Step 8: Optional AI Forecasting using Facebook Prophet
df_prophet = annual_temp.rename(columns={"Year": "ds", "AverageTemperature": "y"})
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], format='%Y')

m = Prophet()
m.fit(df_prophet)

future = m.make_future_dataframe(periods=10, freq='Y')  # Forecast for next 10 years
forecast = m.predict(future)

# Plot the forecast
fig = m.plot(forecast)
plt.title("📈 Temperature Forecast for India")
plt.xlabel("Year")
plt.ylabel("Average Temperature")
plt.grid(True)
plt.show()
