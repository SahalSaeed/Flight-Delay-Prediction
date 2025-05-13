# Flight Delay Prediction

![Flight Delay](https://img.shields.io/badge/Flight-Delay%20Prediction-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Project-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

A full pipeline machine learning project to predict flight delays using data preprocessing, visualization, classification (binary & multi-class), and regression techniques.

## Project Overview

This project aims to develop ML models that predict flight delays based on weather conditions, scheduled departure times, and other flight attributes. It supports binary classification (on-time vs. delayed), multi-class classification (levels of delay), and regression (predicting delay duration in minutes).

---

## Table of Contents

* [Dataset](#dataset)
* [Project Structure](#project-structure)
* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [Methodology](#methodology)
* [Future Work](#future-work)


---

## Dataset

Two datasets are used:

* **flights.csv**: Flight schedule details, carrier info, delay status.
* **weather.csv**: Weather conditions including temperature, wind speed, etc.

These are merged and processed in the notebook to generate a unified training set.

---

## Project Structure

This implementation is notebook-centric:

```
flight-delay-prediction/
├── All-phases.ipynb     # Main Jupyter notebook (includes preprocessing to model evaluation)
├── data/
│   ├── flights.csv
│   └── weather.csv
├── results/
│   ├── plots/           # Contains EDA and correlation plots
│   └── models/          # Saved models and metrics (optional)
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Features

* Data merging, cleaning, and feature extraction in a single notebook
* Binary classification model to detect if a flight will be delayed
* Multi-class model for short, moderate, long delays
* Regression model to predict exact delay minutes
* Visualizations: bar plots, heatmaps, distribution plots
* Evaluation metrics displayed after each training phase

---

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/flight-delay-prediction.git
cd flight-delay-prediction
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

1. Launch the Jupyter Notebook:

```bash
jupyter notebook All-phases.ipynb
```

2. Run the cells in sequence. Key stages include:

   * Data Cleaning & Feature Engineering
   * Visualization & EDA
   * Binary Classification
   * Multi-class Classification
   * Regression Analysis
   * Performance Evaluation

3. Modify and experiment with:

   * Models: `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingRegressor`, etc.
   * Features: Add/remove from engineered dataset
   * Hyperparameters: Via `GridSearchCV`

---

## Methodology

### 1. Data Preprocessing

* Dropping NA, renaming columns
* Merging weather and flight data
* Feature engineering: departure hour, month, day, delay thresholds

### 2. EDA & Visualization

* Delay distribution histograms
* Airport-wise and carrier-wise delay rates
* Heatmaps for feature correlation

### 3. Modeling

* Binary classification (on-time vs. delayed)
* Multi-class classification (no, short, moderate, long delay)
* Regression model (predict delay in minutes)

### 4. Evaluation

* Classification: Accuracy, Precision, Recall, F1
* Regression: MAE, RMSE
* Confusion matrix and class-wise scores

---

## Future Work

* Add deep learning models (e.g., LSTM for time series)
* Use live APIs for real-time weather & flight data
* Build a streamlit or Flask web dashboard
* Predict arrival delays and passenger impact
