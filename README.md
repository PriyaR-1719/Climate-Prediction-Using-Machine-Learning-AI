# Climate-Prediction-Using-Machine-Learning-AI
This project uses historical temperature data to analyze and predict climate trends, specifically average annual temperatures by country using Machine Learning and Prophet (AI-based time series forecasting). The analysis is done on the India dataset, but can be extended to other countries.

 Dataset
Source: Kaggle: Global Land Temperatures By Country
File used: GlobalLandTemperaturesByCountry.csv
Contains average temperature records by country from 1743 to 2013.

Technologies Used
Python
Pandas, Numpy – Data manipulation
Seaborn, Matplotlib – Visualization
Scikit-learn – Machine Learning (Linear Regression)
Facebook Prophet – AI-based forecasting
Google Colab – Development environment

Project Highlights
Preprocessing & cleaning of historical climate data
Visual analysis of temperature trends in India
Linear regression model to predict average temperature
Facebook Prophet model for AI-based future forecasting
Evaluation of model performance using Mean Squared Error

 How to Run
1. Clone the repo
bash
git clone https://github.com/yourusername/climate-prediction-ml.git
cd climate-prediction-ml
2. Upload to Google Colab
Open Google Colab
Upload GlobalLandTemperaturesByCountry.csv
Copy and paste the climate_prediction.ipynb code (from this repo)

Run all cells to train, visualize, and forecast

3. Or Run Locally
bash
pip install pandas numpy matplotlib seaborn scikit-learn prophet
python climate_prediction.py
Sample Output
Annual temperature trends plotted
MSE score for prediction accuracy
Prophet forecast plot for 10 years ahead
 
Future Improvements
Add country selector for dynamic analysis
Use more complex ML models (e.g., Random Forest, XGBoost)
Deploy as a Streamlit web app
Include anomaly detection for extreme weather events

 Author
Priya R.
B.C.A. Final Year | Climate Analytics Enthusiast
Connect on [LinkedIn](url)
