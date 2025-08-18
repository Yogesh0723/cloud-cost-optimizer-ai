import pandas as pd
import numpy as np
import json
from sklearn.ensemble import IsolationForest
from prophet import Prophet

# --------------------
# Step 1: Load & Preprocess Data
# --------------------

def load_data(file_path: str):
    df = pd.read_csv(file_path)
    df["usage_start_time"] = pd.to_datetime(df["usage_start_time"])
    df["date"] = df["usage_start_time"].dt.date
    return df

# --------------------
# Step 2: Idle Resource Detection
# --------------------

def detect_idle_resources(df, threshold=5.0):
    # For simplicity: usage_amount < threshold → idle
    idle_df = df[df["usage_amount"] < threshold]
    return idle_df

# --------------------
# Step 3: Right-Sizing Recommender
# --------------------

def right_sizing(df):
    recs = []
    for _, row in df.iterrows():
        if "n2-standard-8" in row["sku_description"] and row["usage_amount"] < 20:
            recs.append({
                "project_id": row["project_id"],
                "service": row["service"],
                "sku": row["sku_description"],
                "recommendation": "Switch to e2-standard-4",
                "savings_estimate": round(row["cost"] * 0.4, 2)
            })
    return pd.DataFrame(recs)

# --------------------
# Step 4: Anomaly Detection
# --------------------

def detect_anomalies(df):
    daily_cost = df.groupby("date")["cost"].sum().reset_index()
    model = IsolationForest(contamination=0.05, random_state=42)
    daily_cost["anomaly"] = model.fit_predict(daily_cost[["cost"]])
    anomalies = daily_cost[daily_cost["anomaly"] == -1]
    return anomalies

# --------------------
# Step 5: Forecasting Future Spend
# --------------------

def forecast_spend(df, periods=30):
    daily_cost = df.groupby("date")["cost"].sum().reset_index()
    daily_cost.columns = ["ds", "y"]
    model = Prophet()
    model.fit(daily_cost)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

# --------------------
# Step 6: Recommendation Engine
# --------------------

def generate_recommendations(idle_df, rightsize_df, anomaly_df, forecast_df):
    recs = []
    for _, row in idle_df.iterrows():
        recs.append(f"Shut down {row['sku_description']} in {row['project_id']} (idle) → Save approx {row['cost']:.2f}")

    for _, row in rightsize_df.iterrows():
        recs.append(f"Resize {row['sku']} in {row['project_id']} → {row['recommendation']} (Save {row['savings_estimate']})")

    for _, row in anomaly_df.iterrows():
        recs.append(f"⚠️ Anomaly detected on {row['date']} → Spend {row['cost']:.2f}")

    last_forecast = forecast_df.tail(1).iloc[0]
    recs.append(f"Forecast: Next month spend ≈ {last_forecast['yhat']:.2f} (range {last_forecast['yhat_lower']:.2f}-{last_forecast['yhat_upper']:.2f})")
    
    return recs

# --------------------
# Step 7: Save Results to CSV, JSON, TXT
# --------------------

def save_outputs(idle_df, rightsize_df, anomaly_df, forecast_df, recommendations):
    idle_df.to_csv("idle_resources.csv", index=False)
    rightsize_df.to_csv("right_sizing.csv", index=False)
    anomaly_df.to_csv("anomalies.csv", index=False)
    forecast_df.to_csv("forecast.csv", index=False)

    with open("recommendations.json", "w") as f:
        json.dump(recommendations, f, indent=2)

    with open("recommendations.txt", "w") as f:
        for rec in recommendations:
            f.write(rec + "\n")

# --------------------
# Main Pipeline
# --------------------
if __name__ == "__main__":
    df = load_data("gcp_billing_data.csv")

    idle_df = detect_idle_resources(df)
    rightsize_df = right_sizing(df)
    anomaly_df = detect_anomalies(df)
    forecast_df = forecast_spend(df)
    recommendations = generate_recommendations(idle_df, rightsize_df, anomaly_df, forecast_df)

    save_outputs(idle_df, rightsize_df, anomaly_df, forecast_df, recommendations)

    print("✅ Analysis complete. Results saved to CSV, JSON, and TXT files.")
