import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# Load dataset
df = pd.read_csv("data/Delivery_Logistics.csv")


# Convert time columns
df["expected_time_hours"] = (
    df["expected_time_hours"]
    .astype(str)
    .str.extract(r"\.(\d+)$")[0]
    .astype(float)
)


# Convert target
df["delayed"] = df["delayed"].map({
    "no": 0,
    "yes": 1
})


# Features
features = [
    "delivery_partner",
    "package_type",
    "vehicle_type",
    "delivery_mode",
    "region",
    "weather_condition",
    "distance_km",
    "package_weight_kg",
    "expected_time_hours"
]

X = df[features]
y = df["delayed"]


# Same train/test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Load trained models
models = joblib.load("models/trained_models.pkl")


results = []


# Evaluate models
for name, model in models.items():

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probability)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc
    })

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(classification_report(
        y_test,
        y_pred,
        target_names=["On Time", "Delayed"]
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# Model comparison
results_df = pd.DataFrame(results)

print("\n\nMODEL COMPARISON")
print("=" * 80)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

print("\nResults saved to: models/model_comparison.csv")