import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# Load dataset
df = pd.read_csv("data/Delivery_Logistics.csv")


# Convert delivery time columns to numeric hours
df["delivery_time_hours"] = (
    df["delivery_time_hours"]
    .astype(str)
    .str.extract(r"\.(\d+)$")[0]
    .astype(float)
)

df["expected_time_hours"] = (
    df["expected_time_hours"]
    .astype(str)
    .str.extract(r"\.(\d+)$")[0]
    .astype(float)
)


# Convert target to 0/1
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
    "delivery_time_hours",
    "expected_time_hours"
]

X = df[features]
y = df["delayed"]


# Separate categorical and numerical columns
categorical_features = [
    "delivery_partner",
    "package_type",
    "vehicle_type",
    "delivery_mode",
    "region",
    "weather_condition"
]

numerical_features = [
    "distance_km",
    "package_weight_kg",
    "delivery_time_hours",
    "expected_time_hours"
]


# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            SimpleImputer(strategy="median"),
            numerical_features
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        )
    ]
)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Dataset shape:", df.shape)
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("\nTarget distribution:")
print(y.value_counts())