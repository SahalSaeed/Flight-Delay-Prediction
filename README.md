# Flight Delay Prediction

![Flight Delay](https://img.shields.io/badge/Flight-Delay%20Prediction-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Project-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

A comprehensive machine learning project to predict flight delays using weather data, temporal patterns, and flight information.

## Project Overview

This project aims to build predictive models that can accurately forecast flight delays based on various features such as weather conditions, flight schedules, and temporal patterns. The models can classify flights as delayed or on-time (binary classification), categorize delays into different severity levels (multi-class classification), and predict the exact duration of delays (regression).

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Future Work](#future-work)
- [Contact](#contact)

## Dataset

The project utilizes two primary datasets:
- **Flight dataset**: Contains information about flight schedules, actual departure/arrival times, and other flight-related attributes
- **Weather dataset**: Includes weather conditions at different airports, such as temperature, wind speed, and humidity

## Project Structure

```
flight-delay-prediction/
├── data/
│   ├── raw/
│   │   ├── flights.csv
│   │   └── weather.csv
│   └── processed/
│       └── integrated_data.csv
├── notebooks/
│   ├── 1_Data_Preprocessing.ipynb
│   ├── 2_Exploratory_Data_Analysis.ipynb
│   ├── 3_Binary_Classification.ipynb
│   ├── 4_Multiclass_Classification.ipynb
│   └── 5_Regression_Analysis.ipynb
├── src/
│   ├── data_preprocessing/
│   │   ├── __init__.py
│   │   ├── data_integration.py
│   │   ├── data_cleaning.py
│   │   └── feature_engineering.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── exploratory_analysis.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── binary_classification.py
│   │   ├── multiclass_classification.py
│   │   └── regression_models.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── results/
│   ├── visualizations/
│   │   ├── delay_distribution.png
│   │   ├── temporal_analysis.png
│   │   └── correlation_heatmap.png
│   └── model_performance/
│       ├── binary_classification_results.csv
│       ├── multiclass_classification_results.csv
│       └── regression_results.csv
├── requirements.txt
├── setup.py
└── README.md
```

## Features

- **Data Integration**: Combines flight and weather datasets for comprehensive analysis
- **Data Preprocessing**: Handles missing values and formats time fields
- **Feature Engineering**: Calculates departure delays and extracts temporal features
- **Exploratory Data Analysis**: Visualizes delay distributions and analyzes patterns
- **Machine Learning Models**:
  - Binary Classification: Predicts whether a flight will be delayed or on-time
  - Multi-Class Classification: Categorizes delays into different severity levels
  - Regression Analysis: Predicts the exact duration of delays
- **Model Optimization**: Implements hyperparameter tuning and cross-validation techniques
- **Performance Evaluation**: Analyzes model performance using various metrics

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/flight-delay-prediction.git
   cd flight-delay-prediction
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preprocessing

```python
from src.data_preprocessing.data_integration import integrate_datasets
from src.data_preprocessing.data_cleaning import clean_data
from src.data_preprocessing.feature_engineering import engineer_features

# Integrate flight and weather datasets
integrated_data = integrate_datasets('data/raw/flights.csv', 'data/raw/weather.csv')

# Clean the integrated data
cleaned_data = clean_data(integrated_data)

# Engineer features
processed_data = engineer_features(cleaned_data)
```

### Training Models

```python
from src.models.binary_classification import train_binary_classifier
from src.models.multiclass_classification import train_multiclass_classifier
from src.models.regression_models import train_regression_model

# Train binary classification model
binary_model, binary_metrics = train_binary_classifier(X_train, y_train)

# Train multi-class classification model
multiclass_model, multiclass_metrics = train_multiclass_classifier(X_train, y_train)

# Train regression model
regression_model, regression_metrics = train_regression_model(X_train, y_train)
```

### Making Predictions

```python
# Make binary predictions
binary_predictions = binary_model.predict(X_test)

# Make multi-class predictions
multiclass_predictions = multiclass_model.predict(X_test)

# Make regression predictions
regression_predictions = regression_model.predict(X_test)
```

## Methodology

The project follows a comprehensive approach:

### Phase 1: Data Preprocessing and Feature Engineering
- Data integration from multiple sources
- Handling missing values using appropriate imputation techniques
- Standardizing time fields
- Calculating departure delays
- Merging weather data with flight information
- Extracting temporal features (day of week, hour of day, month)

### Phase 2: Exploratory Data Analysis (EDA)
- Visualizing delay distributions
- Analyzing temporal patterns (hourly, daily, monthly)
- Examining category-wise analysis (by airline, airport)
- Performing correlation analysis between weather features and delays

### Phase 3: Analytical and Predictive Tasks
- Binary classification (on-time vs. delayed)
- Multi-class classification (no delay, short delay, moderate delay, long delay)
- Regression analysis to predict exact delay durations

### Phase 4: Model Optimization
- Hyperparameter tuning using grid search and random search
- K-fold cross-validation for robust model evaluation
- Model comparison based on performance metrics

### Phase 5: Model Testing
- Generating predictions on test dataset
- Formatting results according to submission requirements

## Results

### Binary Classification
- **Accuracy**: XX%
- **Precision**: XX%
- **Recall**: XX%
- **F1-Score**: XX%

### Multi-Class Classification
- **Accuracy**: XX%
- **Weighted F1-Score**: XX%
- **Class-wise Performance**:
  - No Delay: XX% accuracy
  - Short Delay: XX% accuracy
  - Moderate Delay: XX% accuracy
  - Long Delay: XX% accuracy

### Regression Analysis
- **Mean Absolute Error (MAE)**: XX minutes
- **Root Mean Square Error (RMSE)**: XX minutes

## Future Work

- Incorporate additional data sources such as air traffic control information
- Develop an interactive web application for real-time flight delay predictions
- Implement advanced deep learning models to capture complex patterns
- Extend the analysis to include arrival delays and connecting flights
- Create an API service for integration with other flight information systems

## Contact

- **Developer**: Sahal Saeed
- **Student ID**: 22i-0476
- **Institution**: FAST NUCES
- **Course**: AI-A
- **Date**: December 12, 2024
- **GitHub**: [Your GitHub Profile](https://github.com/yourusername)
- **Email**: your.email@example.com

---

Feel free to open issues or submit pull requests if you have suggestions for improvements or find any bugs!