# AI FinOps Assistant — Advanced Python Pipeline (Self‑Learning Ready)

"""
End-to-end, production-style Python script that ingests cloud billing data (mock CSV or GCP
BigQuery export), generates rich FinOps insights, and applies advanced ML for anomaly
 detection and forecasting. Outputs results to CSV, JSON, and TXT.

Key Upgrades vs. basic version
- Data enrichment: service breakdowns, month-over-month (MoM) trends, top-N reports
- Idle detection: hybrid (threshold + clustering) with explainable flags
- Right-sizing recommender: rules + price ratios, extensible to regression
- Anomaly detection: Deep Autoencoder (TF/Keras) + IsolationForest ensemble fallback
- Forecasting: LSTM sequence model; Prophet fallback if TensorFlow unavailable
- Self-learning hooks: feedback ledger to retrain rules/thresholds over time
- Savings calculator & action ranking

Note: This script is designed to work on the mock schema used earlier:
  project_id, service, sku_description, usage_start_time, usage_end_time,
  usage_amount, cost

If you have additional telemetry (CPU%, memory%), map those to `utilization_pct`
for more precise right-sizing.
"""

from __future__ import annotations
import os
import json
import argparse
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from pymongo import MongoClient


import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Optional deps
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from keras import layers, models
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
except Exception as e:
    raise RuntimeError("scikit-learn is required: pip install scikit-learn")

# Prophet is optional fallback for forecasting
PROPHET_AVAILABLE = False
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# ---------------------------
# Config & Utilities
# ---------------------------

@dataclass
class Config:
    input_csv: str
    out_dir: str = "outputs"
    idle_usage_threshold: float = 5.0          # usage_amount proxy threshold
    idle_days_window: int = 7                  # consecutive days considered idle
    anomaly_contamination: float = 0.05
    lstm_window: int = 30                      # lookback window for forecasting
    lstm_epochs: int = 20
    lstm_batch_size: int = 32
    seed: int = 42


def ensure_out_dir(out_dir: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, ts)
    os.makedirs(path, exist_ok=True)
    return path


def save_csv(df: pd.DataFrame, path: str):
    if df is None or len(df) == 0:
        pd.DataFrame().to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def save_txt(lines: List[str], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line).rstrip("\n") + "\n")

# ---------------------------
# Step 1 — Load & Preprocess
# ---------------------------

def load_and_preprocess(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.input_csv)
    # Normalize columns
    if "usage_start_time" in df.columns:
        df["usage_start_time"] = pd.to_datetime(df["usage_start_time"])
        df["date"] = df["usage_start_time"].dt.date
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    else:
        raise ValueError("Input CSV must have usage_start_time or date column")

    # Basic hygiene
    for col in ["usage_amount", "cost"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    # Service normalization
    if "service" not in df.columns:
        df["service"] = "Unknown"

    if "project_id" not in df.columns:
        df["project_id"] = "default-project"

    return df


def enrich_insights(df: pd.DataFrame) -> dict:
    """Enrich billing data with more advanced insights."""
    insights = {}

    # ✅ Safe month extraction
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()

    # --- Basic KPIs ---
    total_cost = df["cost"].sum()
    avg_monthly_cost = df.groupby("month")["cost"].sum().mean()
    highest_service = df.groupby("service")["cost"].sum().idxmax()
    highest_service_cost = df.groupby("service")["cost"].sum().max()

    insights["total_cost"] = total_cost
    insights["avg_monthly_cost"] = avg_monthly_cost
    insights["highest_service"] = highest_service
    insights["highest_service_cost"] = highest_service_cost

    # --- Trends ---
    monthly_trends = df.groupby("month")["cost"].sum().reset_index()
    insights["monthly_trends"] = monthly_trends.to_dict(orient="records")

    # --- Service Breakdown ---
    service_breakdown = df.groupby("service")["cost"].sum().reset_index()
    insights["service_breakdown"] = service_breakdown.to_dict(orient="records")

    # --- Project Breakdown ---
    project_breakdown = df.groupby("project_id")["cost"].sum().reset_index()
    insights["project_breakdown"] = project_breakdown.to_dict(orient="records")

    # --- Cost Efficiency Metrics ---
    df["cost_per_unit"] = df["cost"] / df["usage_amount"].replace(0, 1)  # avoid div by zero
    efficiency = df.groupby("service")["cost_per_unit"].mean().reset_index()
    insights["cost_efficiency"] = efficiency.to_dict(orient="records")

    # --- Anomaly Detection (Rule-based) ---
    threshold = avg_monthly_cost * 1.5
    anomalies = monthly_trends[monthly_trends["cost"] > threshold]
    insights["anomalies"] = anomalies.to_dict(orient="records")

    return insights


# -----------------------------------
# Step 2 — Idle Resource Detection
# -----------------------------------

def detect_idle_hybrid(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Hybrid Idle Detector: rule threshold + KMeans clustering.
    We proxy utilization with `usage_amount` at (project, sku) per day.
    """
    # Rule-based: rolling low usage for N days
    usage = df.groupby(["project_id", "sku_description", "date"]).agg(
        usage_amount=("usage_amount", "sum"), cost=("cost", "sum")
    ).reset_index().sort_values(["project_id", "sku_description", "date"]) 

    # Rolling check per resource
    idle_flags = []
    for (proj, sku), g in usage.groupby(["project_id", "sku_description"]):
        g = g.sort_values("date").copy()
        # rolling min over window; mark idle if all below threshold
        g["below_thresh"] = g["usage_amount"] < cfg.idle_usage_threshold
        g["rolling_idle"] = g["below_thresh"].rolling(cfg.idle_days_window, min_periods=cfg.idle_days_window).apply(lambda x: 1 if x.all() else 0)
        # last day indicates current idle status
        current_idle = int(g["rolling_idle"].iloc[-1] == 1)
        idle_flags.append({
            "project_id": proj,
            "sku_description": sku,
            "idle_rule_based": current_idle,
            "recent_avg_usage": g["usage_amount"].tail(cfg.idle_days_window).mean(),
            "recent_avg_cost": g["cost"].tail(cfg.idle_days_window).mean(),
        })
    rb = pd.DataFrame(idle_flags)

    # Clustering on recent average usage
    clean = rb[["recent_avg_usage"]].copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(clean)

    # choose k via simple heuristic
    best_k, best_score = 2, -1
    for k in range(2, min(6, len(rb)) + 1):
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=cfg.seed).fit(X)
            score = silhouette_score(X, km.labels_)
            if score > best_score:
                best_score, best_k = score, k
        except Exception:
            pass
    km = KMeans(n_clusters=max(2, best_k), n_init=10, random_state=cfg.seed).fit(X)
    rb["cluster"] = km.labels_

    # Mark the cluster with lowest mean usage as idle-like
    cluster_means = rb.groupby("cluster")["recent_avg_usage"].mean().sort_values()
    idle_cluster = cluster_means.index[0]
    rb["idle_cluster_based"] = (rb["cluster"] == idle_cluster).astype(int)

    # Final idle
    rb["idle_final"] = ((rb["idle_rule_based"] + rb["idle_cluster_based"]) >= 1).astype(int)

    # Savings estimate heuristic: assume we can stop for 50% of the time next month
    rb["savings_estimate"] = (rb["recent_avg_cost"].fillna(0) * 30 * 0.5).round(2)

    return rb.sort_values(["idle_final", "savings_estimate"], ascending=[False, False])

# -----------------------------------
# Step 3 — Right-Sizing Recommender
# -----------------------------------

PRICE_RATIO = {
    # very rough cost ratios; replace with live price catalog later
    "n2-standard-8": 1.0,
    "n2-standard-4": 0.6,
    "e2-standard-4": 0.45,
    "e2-standard-2": 0.25,
}

SKU_DOWNSIZE_CHAIN = [
    "n2-standard-8",
    "n2-standard-4",
    "e2-standard-4",
    "e2-standard-2",
]

def right_sizing(df: pd.DataFrame) -> pd.DataFrame:
    # Aggregate by resource to compute avg usage & cost
    agg = df.groupby(["project_id", "service", "sku_description"]).agg(
        avg_usage=("usage_amount", "mean"), avg_cost=("cost", "mean")
    ).reset_index()

    recs = []
    for _, r in agg.iterrows():
        sku = str(r["sku_description"]) if pd.notnull(r["sku_description"]) else ""
        if sku not in SKU_DOWNSIZE_CHAIN:
            continue
        idx = SKU_DOWNSIZE_CHAIN.index(sku)
        # naive utilization proxy: if usage is in lowest quartile for that SKU, propose downsize
        # In real system, map usage→CPU% & cap of SKU
        if r["avg_usage"] <= agg[agg["sku_description"] == sku]["avg_usage"].quantile(0.25):
            if idx + 1 < len(SKU_DOWNSIZE_CHAIN):
                target = SKU_DOWNSIZE_CHAIN[idx + 1]
                src_price = PRICE_RATIO.get(sku, 1.0)
                tgt_price = PRICE_RATIO.get(target, src_price)
                savings = max(0.0, (src_price - tgt_price) / max(src_price, 1e-6))
                recs.append({
                    "project_id": r["project_id"],
                    "service": r["service"],
                    "from_sku": sku,
                    "to_sku": target,
                    "avg_usage": round(float(r["avg_usage"]), 2),
                    "avg_cost": round(float(r["avg_cost"]), 2),
                    "estimated_savings_pct": round(savings * 100, 1),
                    "rationale": "Low relative usage for SKU; target cheaper class",
                })
    return pd.DataFrame(recs).sort_values("estimated_savings_pct", ascending=False)

# -----------------------------------
# Step 4 — Anomaly Detection (Ensemble)
# -----------------------------------

def detect_anomalies_ensemble(daily_cost: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = daily_cost.copy()
    out.columns = ["date", "cost"]

    # Isolation Forest
    iso = IsolationForest(contamination=cfg.anomaly_contamination, random_state=cfg.seed)
    iso_labels = iso.fit_predict(out[["cost"]])
    out["iso_outlier"] = (iso_labels == -1).astype(int)

    # Autoencoder (if TF available & enough points)
    if TF_AVAILABLE and len(out) >= 60:
        X = out[["cost"]].values.astype("float32")
        # scale to 0-1
        s_min, s_max = X.min(), X.max()
        rng = max(s_max - s_min, 1e-6)
        Xs = (X - s_min) / rng

        model = models.Sequential([
            layers.Input(shape=(1,)),
            layers.Dense(8, activation="relu"),
            layers.Dense(2, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(Xs, Xs, epochs=100, batch_size=16, verbose=0)
        recon = model.predict(Xs, verbose=0)
        mse = ((Xs - recon) ** 2).reshape(-1)
        thresh = np.percentile(mse, 95)
        out["ae_outlier"] = (mse > thresh).astype(int)
    else:
        out["ae_outlier"] = 0

    out["anomaly_score"] = out[["iso_outlier", "ae_outlier"]].sum(axis=1)
    anomalies = out[out["anomaly_score"] > 0].copy()
    return anomalies.sort_values("date")

# -----------------------------------
# Step 5 — Forecasting (LSTM with fallback)
# -----------------------------------

def create_sequences(arr: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(arr) - window):
        X.append(arr[i:i+window])
        y.append(arr[i+window])
    X = np.array(X).astype("float32")
    y = np.array(y).astype("float32")
    return X[..., None], y  # add feature dim


def forecast_lstm(daily_cost: pd.DataFrame, cfg: Config, horizon: int = 30) -> pd.DataFrame:
    series = daily_cost.sort_values("date")["cost"].values.astype("float32")
    # scale 0-1
    s_min, s_max = series.min(), series.max()
    rng = max(s_max - s_min, 1e-6)
    s = (series - s_min) / rng

    if len(s) <= cfg.lstm_window + 1:
        raise ValueError("Not enough data for LSTM window")

    X, y = create_sequences(s, cfg.lstm_window)

    model = models.Sequential([
        layers.Input(shape=(cfg.lstm_window, 1)),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=cfg.lstm_epochs, batch_size=cfg.lstm_batch_size, verbose=0)

    # iterative forecast
    window_seq = s[-cfg.lstm_window:].copy()
    preds = []
    for _ in range(horizon):
        x = window_seq.reshape(1, cfg.lstm_window, 1)
        p = model.predict(x, verbose=0)[0, 0]
        preds.append(p)
        window_seq = np.concatenate([window_seq[1:], [p]])

    # rescale
    preds = np.array(preds) * rng + s_min

    start_date = pd.to_datetime(daily_cost["date"].max()) + pd.Timedelta(days=1)
    dates = pd.date_range(start=start_date, periods=horizon, freq="D")
    return pd.DataFrame({"date": dates.date, "forecast": preds})


def forecast_with_fallback(daily_cost: pd.DataFrame, cfg: Config, horizon: int = 30) -> pd.DataFrame:
    if TF_AVAILABLE:
        try:
            return forecast_lstm(daily_cost, cfg, horizon)
        except Exception:
            pass
    if PROPHET_AVAILABLE:
        tmp = daily_cost.copy()
        tmp.columns = ["ds", "y"]
        m = Prophet()
        m.fit(tmp)
        future = m.make_future_dataframe(periods=horizon)
        fc = m.predict(future)[["ds", "yhat"]].tail(horizon)
        fc.columns = ["date", "forecast"]
        fc["date"] = pd.to_datetime(fc["date"]).dt.date
        return fc
    # naive fallback: last value persistence
    last = float(daily_cost["cost"].iloc[-1]) if len(daily_cost) else 0.0
    start_date = pd.to_datetime(daily_cost["date"].max()) + pd.Timedelta(days=1)
    dates = pd.date_range(start=start_date, periods=horizon, freq="D")
    return pd.DataFrame({"date": dates.date, "forecast": [last]*horizon})

# -----------------------------------
# Step 6 — Recommendation Engine & Ranking
# -----------------------------------

def build_recommendations(idle_df: pd.DataFrame, rs_df: pd.DataFrame, anomalies: pd.DataFrame, forecast_df: pd.DataFrame) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []

    # Idle actions
    for _, r in idle_df[idle_df["idle_final"] == 1].iterrows():
        recs.append({
            "type": "stop_idle",
            "project_id": r["project_id"],
            "resource": r["sku_description"],
            "saving_estimate_month": float(r.get("savings_estimate", 0)),
            "explanation": f"Idle by rules/clustering. Recent avg usage={r['recent_avg_usage']:.2f}",
            "risk": "low",
        })

    # Right-size actions
    for _, r in rs_df.iterrows():
        recs.append({
            "type": "rightsize",
            "project_id": r["project_id"],
            "resource": f"{r['from_sku']}→{r['to_sku']}",
            "saving_estimate_pct": float(r["estimated_savings_pct"]),
            "explanation": r["rationale"],
            "risk": "medium",
        })

    # Anomaly alerts (not savings but guardrails)
    for _, r in anomalies.iterrows():
        recs.append({
            "type": "anomaly_alert",
            "date": str(r["date"]),
            "spend": float(r["cost"]),
            "explanation": "Spend outlier (IForest/AE ensemble)",
            "risk": "n/a",
        })

    # Forecast summary
    month_forecast = float(forecast_df["forecast"].sum())
    recs.append({
        "type": "forecast_summary",
        "horizon_days": int(len(forecast_df)),
        "projected_spend_sum": month_forecast,
        "explanation": "Projected next-period spend",
    })

    # Ranking: prioritize direct savings first by absolute savings, then pct
    def score(r):
        if r["type"] == "stop_idle":
            return -r.get("saving_estimate_month", 0)
        if r["type"] == "rightsize":
            return -r.get("saving_estimate_pct", 0)
        return 0

    recs = sorted(recs, key=score)
    return recs

# -----------------------------------
# Step 7 — Outputs & Feedback Hooks
# -----------------------------------

def write_all_outputs(out_dir: str,
                      insights: Dict[str, pd.DataFrame],
                      idle_df: pd.DataFrame,
                      rs_df: pd.DataFrame,
                      anomalies: pd.DataFrame,
                      forecast_df: pd.DataFrame,
                      recs: List[Dict[str, Any]],
                      savings_proj: pd.DataFrame):

    # ✅ Safely save insights only if they exist
    if "daily" in insights:
        save_csv(insights["daily"], os.path.join(out_dir, "daily_cost.csv"))
    if "monthly" in insights:
        save_csv(insights["monthly"], os.path.join(out_dir, "monthly_cost.csv"))
    if "svc_monthly" in insights:
        save_csv(insights["svc_monthly"], os.path.join(out_dir, "service_monthly_cost.csv"))
    if "top_services" in insights:
        save_csv(insights["top_services"], os.path.join(out_dir, "top_services.csv"))
    if "top_projects" in insights:
        save_csv(insights["top_projects"], os.path.join(out_dir, "top_projects.csv"))

    # Models
    save_csv(idle_df, os.path.join(out_dir, "idle_candidates.csv"))
    save_csv(rs_df, os.path.join(out_dir, "rightsizing_recommendations.csv"))
    save_csv(anomalies, os.path.join(out_dir, "anomalies.csv"))
    save_csv(forecast_df, os.path.join(out_dir, "forecast.csv"))
    save_csv(savings_proj, os.path.join(out_dir, "savings_projection.csv")) 
    # Recommendations
    save_json(recs, os.path.join(out_dir, "recommendations.json"))
    save_txt([json.dumps(x) for x in recs], os.path.join(out_dir, "recommendations.txt"))

    # Summary text
    summary_lines = [
        "AI FinOps Assistant — Run Summary",
        f"Daily points: {len(insights['daily']) if 'daily' in insights else 0}",
        f"Monthly points: {len(insights['monthly']) if 'monthly' in insights else 0}",
        f"Idle candidates: {int(idle_df['idle_final'].sum()) if 'idle_final' in idle_df.columns else 0}",
        f"Right-sizing recs: {len(rs_df)}",
        f"Anomalies: {len(anomalies)}",
        f"Forecast horizon days: {len(forecast_df)}",
    ]
    save_txt(summary_lines, os.path.join(out_dir, "summary.txt"))


def build_savings_projection(forecast_df, idle_df, rs_df, out_dir):
    # Total predicted spend
    predicted = float(forecast_df["forecast"].sum())

    # Potential savings
    idle_savings = float(idle_df[idle_df["idle_final"] == 1]["savings_estimate"].sum())
    rs_savings_pct = float(rs_df["estimated_savings_pct"].sum()) / 100.0
    rs_savings = predicted * rs_savings_pct

    total_savings = idle_savings + rs_savings
    optimized = max(0.0, predicted - total_savings)

    month = pd.to_datetime(forecast_df["date"].iloc[0]).strftime("%Y-%m")
    df = pd.DataFrame([{
        "month": month,
        "predicted_spend": round(predicted, 2),
        "potential_savings": round(total_savings, 2),
        "optimized_spend": round(optimized, 2)
    }])

    save_csv(df, os.path.join(out_dir, "savings_projection.csv"))
    return df


# Feedback ledger: append accepted/rejected actions for self-learning

def log_feedback(out_dir: str, action_id: str, decision: str, metadata: Optional[Dict[str, Any]] = None):
    path = os.path.join(out_dir, "feedback_ledger.jsonl")
    entry = {"ts": datetime.now(timezone.utc).isoformat(), "id": action_id, "decision": decision, "meta": metadata or {}}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def save_to_mongo(db_url: str, db_name: str, data_dict: Dict[str, pd.DataFrame], recs: List[Dict[str, Any]]):
    """Save all generated dataframes and recommendations into MongoDB"""
    client = MongoClient(db_url)
    db = client[db_name]

    # Save each dataframe as a collection
    for key, df in data_dict.items():
        if df is None or len(df) == 0:
            continue
        records = df.to_dict(orient="records")
        if records:
            db[key].delete_many({})  # clean old
            db[key].insert_many(records)

    # Save recommendations
    if recs:
        db["recommendations"].delete_many({})
        db["recommendations"].insert_many(recs)

    print(f"✅ Data saved into MongoDB Database: {db_name}")


# ---------------------------
# Main
# ---------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="AI FinOps Assistant — Advanced")
    parser.add_argument("--input_csv", required=True, help="Path to billing CSV")
    parser.add_argument("--out_dir", default="outputs", help="Output base directory")
    parser.add_argument("--idle_threshold", type=float, default=5.0, help="Idle usage threshold (proxy)")
    parser.add_argument("--idle_window", type=int, default=7, help="Idle rolling window (days)")
    parser.add_argument("--horizon", type=int, default=30, help="Forecast horizon (days)")
    parser.add_argument("--lstm_epochs", type=int, default=20)
    args = parser.parse_args(argv)

    cfg = Config(
        input_csv=args.input_csv,
        out_dir=args.out_dir,
        idle_usage_threshold=args.idle_threshold,
        idle_days_window=args.idle_window,
        lstm_epochs=args.lstm_epochs,
    )

    out_dir_ts = ensure_out_dir(cfg.out_dir)

    print("[1/7] Loading & preprocessing…")
    df = load_and_preprocess(cfg)

    print("[1b] Building insights…")
    insights = enrich_insights(df)

    # ✅ Build daily/monthly dataframes explicitly
    daily_cost = df.groupby("date")["cost"].sum().reset_index()
    daily_cost.columns = ["date", "cost"]
    monthly_cost = df.groupby("month")["cost"].sum().reset_index()

    # Attach to insights dict for downstream use
    insights["daily"] = daily_cost
    insights["monthly"] = monthly_cost

    print("[2/7] Detecting idle resources (hybrid)…")
    idle_df = detect_idle_hybrid(df, cfg)

    print("[3/7] Generating right-sizing recommendations…")
    rs_df = right_sizing(df)

    print("[4/7] Detecting anomalies (ensemble)…")
    anomalies = detect_anomalies_ensemble(daily_cost, cfg)

    print("[5/7] Forecasting spend… (TF available = {} )".format(TF_AVAILABLE))
    forecast_df = forecast_with_fallback(daily_cost, cfg, horizon=args.horizon)

    print("[6/7] Compiling recommendations…")
    recs = build_recommendations(idle_df, rs_df, anomalies, forecast_df)

    print("[7b] Building savings projection…")
    savings_proj = build_savings_projection(forecast_df, idle_df, rs_df, out_dir_ts)

    print("[7/7] Writing outputs… →", out_dir_ts)
    write_all_outputs(out_dir_ts, insights, idle_df, rs_df, anomalies, forecast_df, recs, savings_proj)
    
    # -----------------------------
    # Save to MongoDB
    # -----------------------------
    mongo_url = "mongodb+srv://yogupatil135:Yogesh%402001@laxmidevi.liw3ism.mongodb.net/?retryWrites=true&w=majority&appName=laxmiDevi"
    db_name = "finops_db"

    # Prepare dict of datasets
    datasets = {
        "daily_cost": insights.get("daily"),
        "monthly_cost": insights.get("monthly"),
        "service_monthly_cost": insights.get("svc_monthly"),
        "top_services": insights.get("top_services"),
        "top_projects": insights.get("top_projects"),
        "idle_candidates": idle_df,
        "rightsizing_recommendations": rs_df,
        "anomalies": anomalies,
        "forecast": forecast_df,
        "savings_projection": savings_proj,
    }

    save_to_mongo(mongo_url, db_name, datasets, recs)


    print("[8/8] Running self-learning feedback…")
    log_feedback(out_dir_ts, "idle_stop", "accepted", {"idle_count": int(idle_df["idle_final"].sum())})
    log_feedback(out_dir_ts, "rightsize", "accepted", {"count": len(rs_df)})

    print("[9/9] Running self-learning feedback…")
    log_feedback(out_dir_ts, "forecast_summary", "accepted", {"horizon_days": len(forecast_df)})
    log_feedback(out_dir_ts, "anomaly_alert", "accepted", {"count": len(anomalies)})
    log_feedback(out_dir_ts, "rightsize", "accepted", {"count": len(rs_df)})
    log_feedback(out_dir_ts, "idle_stop", "accepted", {"idle_count": int(idle_df["idle_final"].sum())})
    log_feedback(out_dir_ts, "savings_projection", "accepted", {"predicted_spend": float(savings_proj["predicted_spend"].sum())})

    print("✅ Done. Artifacts written to:", out_dir_ts)
    for f in sorted(os.listdir(out_dir_ts)):
        print("  -", f)

if __name__ == "__main__":
    main()


# python cloud_cost_optimizer.py --input_csv gcp_billing_data.csv --horizon 30 --lstm_epochs 20
