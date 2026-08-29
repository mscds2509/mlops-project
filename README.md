# Delivery Delay Prediction using Machine Learning

## Overview

This project uses machine learning to predict whether a logistics delivery is likely to be delayed based on information available before or during dispatch.

The project compares multiple classification models and uses **XGBoost** as the final prediction model.

A **Delivery Risk Level** feature has also been added to provide an easier-to-understand interpretation of the model's predicted delay probability.

The system provides:

- **On Time / Delayed** prediction
- **Probability of delay**
- **Delivery Risk Level: LOW / MEDIUM / HIGH**

---

## Problem Statement

Delivery delays can affect customer satisfaction, logistics planning, and operational efficiency.

The goal of this project is to predict whether a delivery will be:

- **On Time**
- **Delayed**

using information such as:

- Delivery partner
- Package type
- Vehicle type
- Delivery mode
- Region
- Weather condition
- Distance
- Package weight
- Expected delivery time

The model can help identify deliveries that may require additional attention before a delay occurs.

---

## Dataset

The dataset contains **25,000 delivery records** and **15 original features**.

Important features include:

| Feature | Description |
|---|---|
| `delivery_partner` | Logistics/delivery company |
| `package_type` | Type of package |
| `vehicle_type` | Vehicle used for delivery |
| `delivery_mode` | Same-day, express, or two-day delivery |
| `region` | Delivery region |
| `weather_condition` | Weather condition |
| `distance_km` | Delivery distance |
| `package_weight_kg` | Package weight |
| `expected_time_hours` | Expected delivery time |

### Target Variable

`delayed`

The target is encoded as:

- `0` → On Time
- `1` → Delayed

---

## Data Leakage Handling

During data analysis, `delivery_time_hours` was found to strongly reveal the target variable.

Using this feature would result in **target leakage** because the actual delivery time would only be known after or during the completion of the delivery.

Therefore, `delivery_time_hours` was removed from the final prediction features.

This creates a more realistic **pre-delivery prediction problem**, where the model only uses information that could reasonably be available before the delivery outcome is known.

---

## Machine Learning Models

Three classification models were evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

### Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 89.40% | 79.43% | 81.33% | 80.37% | 96.68% |
| Random Forest | 89.24% | 79.48% | 80.43% | 79.96% | 96.49% |
| **XGBoost** | **89.56%** | 79.42% | **82.16%** | **80.77%** | 96.61% |

XGBoost was selected as the final model because it achieved the highest:

- Accuracy
- Recall
- F1-score

Recall is particularly useful for this problem because identifying actual delayed deliveries is important from a logistics perspective.

---

## Delivery Risk Level Feature

In addition to predicting whether a delivery will be delayed, the project converts the model's delay probability into an easy-to-understand risk level.

The risk levels are:

| Delay Probability | Risk Level |
|---:|---|
| `< 40%` | LOW |
| `40% – <70%` | MEDIUM |
| `>= 70%` | HIGH |

For example:

```text
Prediction: DELAYED
Probability of delay: 95.34%
Risk Level: HIGH