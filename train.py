import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load dataset
df = pd.read_csv("AirPassengers.csv")

# Rename columns for clarity
df.columns = ["Month", "Passengers"]

# Create time index
df["t"] = np.arange(len(df))

X = df[["t"]]
y = df["Passengers"]

# MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("AirPassengers-Regression")

with mlflow.start_run():
    model = LinearRegression()
    model.fit(X, y)

    preds = model.predict(X)
    mse = mean_squared_error(y, preds)

    mlflow.log_metric("mse", mse)
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.sklearn.log_model(model, "model")

    print("Training completed")
    print("MSE:", mse)
