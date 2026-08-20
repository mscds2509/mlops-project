# Delivery Delay Prediction using Machine Learning

## Overview

This project predicts whether a logistics delivery is likely to be delayed based on delivery and operational information available before/during dispatch.

The project compares multiple machine learning classification models and uses XGBoost as the final model.

## Problem Statement

Delivery delays can affect customer satisfaction, logistics planning, and operational efficiency.

The goal of this project is to predict:

- **On Time**
- **Delayed**

using information such as delivery partner, package type, vehicle type, delivery mode, region, weather, distance, package weight, and expected delivery time.

## Dataset

The dataset contains 25,000 delivery records and 15 original features.

Important features include:

- Delivery partner
- Package type
- Vehicle type
- Delivery mode
- Region
- Weather condition
- Distance
- Package weight
- Expected delivery time

Target variable:

- `delayed`

## Data Leakage Handling

During data analysis, `delivery_time_hours` was found to strongly reveal the target variable.

Using this feature would result in target leakage because actual delivery time is not available when making a pre-delivery prediction.

Therefore, `delivery_time_hours` was removed from the final model.

This resulted in a more realistic prediction problem.

## Models

Three classification models were evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 89.40% | 79.43% | 81.33% | 80.37% | 96.68% |
| Random Forest | 89.24% | 79.48% | 80.43% | 79.96% | 96.49% |
| **XGBoost** | **89.56%** | 79.42% | **82.16%** | **80.77%** | 96.61% |

XGBoost was selected as the final model because it achieved the highest accuracy, recall, and F1-score.

## Project Structure

```text
mlops-project/
│
├── data/
│   └── Delivery_Logistics.csv
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── model_comparison.csv
│
├── .gitignore
├── requirements.txt
└── README.md