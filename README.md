# AI-Based-Agricultural-Input-Cost-Forecasting
Forecast cost of agricultural inputs to aid financial planning for farmers and agri-businesses.
# AI-Based Agricultural Input Cost Forecasting

## Project Overview

This project predicts Urea fertilizer prices using Machine Learning techniques. The model utilizes historical fertilizer prices, natural gas prices, crude oil prices, and lag features to forecast future fertilizer costs.

## Problem Statement

Agricultural input costs significantly impact farming profitability. Predicting fertilizer prices in advance can help farmers, policymakers, and businesses make informed decisions.

## Objective

To develop a machine learning model capable of forecasting Urea fertilizer prices based on historical market and energy price data.

## Dataset Features

### Input Features

* NaturalGasPrice
* CrudeOilPrice
* Year
* Month
* Urea_Lag1
* Urea_Lag3

### Target Variable

* UreaPrice

## Technologies Used

* Python
* Google Colab
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Streamlit

## Machine Learning Models

1. Linear Regression
2. Random Forest Regressor
3. XGBoost Regressor

## Model Performance

| Model             | MAE   | RMSE  | R² Score |
| ----------------- | ----- | ----- | -------- |
| Linear Regression | 17.09 | 37.81 | 0.949    |
| Random Forest     | 21.08 | 50.00 | 0.911    |
| XGBoost           | 22.57 | 57.91 | 0.880    |

### Best Model

Linear Regression achieved the highest R² score and lowest prediction error, making it the selected model for deployment.

## Example Prediction

Input:

* NaturalGasPrice = 3.5
* CrudeOilPrice = 80
* Year = 2026
* Month = 7
* Urea_Lag1 = 450
* Urea_Lag3 = 430

Predicted Urea Price:

* 402.97

## Project Workflow

Dataset Collection
→ Data Preprocessing
→ Exploratory Data Analysis (EDA)
→ Feature Selection
→ Train-Test Split
→ Model Training
→ Model Evaluation
→ Forecasting
→ Streamlit Deployment

## Repository Structure

```text
├── Agricultural_Input_Cost_Forecasting.ipynb
├── urea_price_model.pkl
├── final_fertilizer_dataset.csv
├── app.py
├── requirements.txt
└── README.md
```

## Future Scope

* Include additional fertilizer categories.
* Integrate live commodity price APIs.
* Develop advanced time-series forecasting models.
* Deploy the system on cloud platforms.

## Authors

B.Tech Project Team

AI-Based Agricultural Input Cost Forecasting
