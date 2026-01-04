import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

df = pd.read_csv("AirPassengers.csv")
df.columns = ["Month", "Passengers"]

df["t"] = np.arange(len(df))
X = df[["t"]]
y = df["Passengers"]

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
