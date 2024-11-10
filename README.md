# Oil Price Insights - Forecasting and Explainability

## Project Overview
This project focuses on analyzing the relationship between significant global events and Brent oil prices, utilizing advanced time series modeling and machine learning techniques. The goal is to predict oil price movements and explain the factors contributing to those predictions. The project is structured into four main tasks:

- **Task 2:** Advanced analysis and exploration of historical Brent oil prices.
- **Task 3:** Forecasting Brent oil prices using ARIMA, ETS, and LSTM models.
- **Task 4:** Model explainability using SHAP to interpret predictions from the LSTM model.

## Table of Contents
- [Task 2: Advanced Analysis of Brent Oil Prices](#task-2-advanced-analysis-of-brent-oil-prices)
  - [Data Exploration (EDA)](#data-exploration-eda)
  - [Statistical Summaries](#statistical-summaries)
  - [Event Impact Analysis](#event-impact-analysis)
- [Task 3: Forecasting Brent Oil Prices](#task-3-forecasting-brent-oil-prices)
  - [Data Preparation and Stationarity Check](#data-preparation-and-stationarity-check)
  - [ARIMA Model for Forecasting](#arima-model-for-forecasting)
  - [ETS Model for Forecasting](#ets-model-for-forecasting)
  - [LSTM Model for Forecasting](#lstm-model-for-forecasting)
  - [Forecast Evaluation and Comparison](#forecast-evaluation-and-comparison)
- [Task 4: Model Explainability](#task-4-model-explainability)
  - [SHAP Analysis for LSTM Predictions](#shap-analysis-for-lstm-predictions)
  - [Feature Contribution Interpretation](#feature-contribution-interpretation)
  - [Insights and Visualization](#insights-and-visualization)

## Project Structure

├── requirements.txt           # Required dependencies for the project
├── README.md                  # Project documentation
├── src/                       # Source code for model building and analysis
│   ├── __init__.py
│   └── explainability.py      # SHAP analysis for model explainability
├── models/                    # Saved models
│   └── lstm_brent_oil_price_model.h5  # Trained LSTM model
├── notebooks/                 # Jupyter notebooks for analysis
│   ├── data_analysis_task2.ipynb   # Task 2: Exploratory Data Analysis
│   ├── forecasting_task3.ipynb    # Task 3: Forecasting Models (ARIMA, ETS, LSTM)
│   └── explainability_task4.ipynb  # Task 4: SHAP Explainability
├── Data/                      # Raw and processed data files
│   └── BrentOilPrice.csv      # Historical Brent Oil Price Data
└── scripts/                   # Helper scripts for data preprocessing and model evaluation
    └── data_preprocessing.py  # Data cleaning and preprocessing script

## Requirements
- Python 3.6+
- TensorFlow 2.x (for LSTM)
- Keras (for LSTM model)
- statsmodels (for ARIMA and ETS models)
- pmdarima (optional for auto ARIMA, not used in case of issues)
- shap (for SHAP analysis)
- matplotlib, seaborn, pandas, numpy (for data manipulation and visualization)

Install the necessary dependencies by running:

pip install -r requirements.txt

## Task Descriptions

### Task 2: Advanced Analysis of Brent Oil Prices
In this task, we perform exploratory data analysis (EDA) on historical Brent oil prices, focusing on:
- **Statistical analysis** to summarize key trends and volatility in the data.
- **Event impact analysis** to observe how specific geopolitical or economic events influenced oil prices over time.

### Task 3: Forecasting Brent Oil Prices
In this task, we build forecasting models to predict future Brent oil prices, including:
- **ARIMA (AutoRegressive Integrated Moving Average):** A time series model suitable for forecasting based on historical data.
- **ETS (Exponential Smoothing):** A model designed to capture trend and seasonality in the time series data.
- **LSTM (Long Short-Term Memory):** A deep learning model used to capture long-range dependencies in the data.

We evaluate each model's performance using **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)** and compare their forecasting results.

### Task 4: Model Explainability
Using **SHAP (SHapley Additive exPlanations)**, we provide insights into the LSTM model’s predictions. The goal is to understand how each feature (or time step in the case of LSTM) contributes to the final prediction. We visualize these contributions using SHAP summary plots and dependence plots.

## How to Use

### Data Preprocessing
Ensure your dataset (Brent oil prices) is placed in the `Data/` folder. The data should be a CSV with columns: Date and Price.

### Task 2: Data Exploration
Open the `data_analysis_task2.ipynb` notebook in the `notebooks/` folder to perform the exploratory data analysis.

### Task 3: Forecasting
Open the `forecasting_task3.ipynb` notebook to explore the ARIMA, ETS, and LSTM models, train them, and compare their forecasting results.

### Task 4: Model Explainability
Open the `explainability_task4.ipynb` notebook to analyze the LSTM model using SHAP, visualize feature impacts, and interpret the model’s predictions.

## Model Saving and Loading
After training the LSTM model in Task 3, you can save it as follows:

model.save('models/lstm_brent_oil_price_model.h5')