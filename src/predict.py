import pandas as pd
import joblib


# Load trained models
models = joblib.load("models/trained_models.pkl")

# Use XGBoost as the final model
model = models["XGBoost"]


print("\nDelivery Delay Prediction")
print("=" * 40)


# Collect delivery information
delivery_partner = input(
    "Delivery partner: "
)

package_type = input(
    "Package type: "
)

vehicle_type = input(
    "Vehicle type: "
)

delivery_mode = input(
    "Delivery mode: "
)

region = input(
    "Region: "
)

weather_condition = input(
    "Weather condition: "
)

distance_km = float(
    input("Distance (km): ")
)

package_weight_kg = float(
    input("Package weight (kg): ")
)

expected_time_hours = float(
    input("Expected delivery time (hours): ")
)


# Create input dataframe
input_data = pd.DataFrame([{
    "delivery_partner": delivery_partner,
    "package_type": package_type,
    "vehicle_type": vehicle_type,
    "delivery_mode": delivery_mode,
    "region": region,
    "weather_condition": weather_condition,
    "distance_km": distance_km,
    "package_weight_kg": package_weight_kg,
    "expected_time_hours": expected_time_hours
}])


# Make prediction
prediction = model.predict(input_data)[0]

probability = model.predict_proba(input_data)[0][1]
if probability < 0.40:
    risk_level = "LOW"
elif probability < 0.70:
    risk_level = "MEDIUM"
else:
    risk_level = "HIGH"

# Display results
print("\n" + "=" * 40)

if prediction == 1:
    print("Prediction: DELAYED")
else:
    print("Prediction: ON TIME")

print(f"Probability of delay: {probability * 100:.2f}%")
print(f"Risk Level: {risk_level}")

print("=" * 40)