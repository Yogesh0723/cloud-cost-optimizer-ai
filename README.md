🚀 AI-Powered Cloud Cost Optimizer

This project helps IT and finance teams reduce cloud cost chaos by using AI/ML on billing data.
It analyzes cloud billing exports (GCP BigQuery, AWS Cost Explorer, Azure Cost Management) to
detect idle resources, recommend right-sizing, forecast future spend, and flag anomalies.

✨ Features
- Automated billing data ingestion (CSV or BigQuery)
- Idle VM / cluster detection using ML + thresholds
- Right-sizing recommendations for compute/database resources
- Anomaly detection for sudden spend spikes
- Forecasting of monthly cloud spend using time series models
- Dashboard with AI-suggested savings and "what-if" simulations
- (Optional) Self-healing mode to auto-stop idle test/dev resources

🛠 Tech Stack
- Backend: Python (FastAPI)
- AI/ML: scikit-learn, Prophet, PyTorch/TensorFlow (for anomaly & forecasting)
- Data: BigQuery / CSV exports
- Frontend: React + Plotly/Recharts
- Deployment: Docker + Cloud Run (GCP)

🌍 Impact
- Save 20–40% on cloud bills without workload changes
- Prevent surprise cost spikes
- Provide CFO + IT transparency on spend
- Scales from a single project to multi-cloud environments



🔎 Files & Their Meaning
    1. daily_cost.csv
    Columns: date, cost
    Shows total spend per day.
    Useful for spotting spikes and feeding anomaly detection / forecasting.

2. monthly_cost.csv
    Columns: month, cost, mom_change_pct
    Shows total spend per month.
    mom_change_pct: Month-over-month % increase/decrease in cost.

3. service_monthly_cost.csv
    Columns: month, service, cost
    Breakdown of how much each service (Compute, Storage, BigQuery, etc.) costs per month.

4. top_services.csv
    Columns: service, cost
    Ranks services by total cost contribution (e.g., “Compute Engine = 60% of spend”).

5. top_projects.csv
    Columns: project_id, cost

Which projects are consuming the most cloud spend.


⚡ Optimization & Detection

6. idle_candidates.csv
    Shows resources that are likely idle (under-utilized).
    Important Columns:
    idle_rule_based: Flagged idle if < threshold usage for X days in a row.
    idle_cluster_based: Flagged idle by ML clustering (KMeans).
    idle_final: 1 = idle confirmed by either rule or ML.
    savings_estimate: How much you’d save/month if you shut it down.

👉 Interpretation: If idle_final=1 and savings_estimate=₹5000, that’s a candidate for shutdown or schedule-based auto-stop.

7. rightsizing_recommendations.csv
    Suggests smaller/cheaper VM types for over-provisioned resources.
    Important Columns:
    from_sku → to_sku: Suggested downgrade (e.g., n2-standard-8 → e2-standard-4).
    estimated_savings_pct: % savings if applied.
    rationale: Why this recommendation was made.

👉 Interpretation: If VM X is 40% over-provisioned, downsizing could save ~40%.

8. anomalies.csv
    Spend anomalies flagged by ML.
    Important Columns:
    iso_outlier: Outlier flagged by Isolation Forest (statistical method).
    ae_outlier: Outlier flagged by Autoencoder (deep learning).
    anomaly_score: Sum of the two methods (0 = normal, 1 = anomaly by one, 2 = anomaly by both).
    A row here means: “On date X, cost = ₹Y, flagged as abnormal compared to history”.

👉 Interpretation: If cost suddenly jumps 3x in a day, this file catches it.

9. forecast.csv
    Prediction of future daily costs (next 30 days by default).
    Columns: date, forecast

Values are in the same unit as cost (e.g., ₹ per day).

👉 Interpretation:

You can sum this up to get next month’s predicted spend.

10. recommendations.json / recommendations.txt
    The combined AI assistant’s advice.

Types of entries:
    stop_idle: Shut down idle resources.
    rightsize: Switch to smaller instances.
    anomaly_alert: “Hey, check spend spike on this date.”
    forecast_summary: “Expected total spend next 30 days = ₹XYZ.”

👉 Interpretation: This is the action list for FinOps + IT teams.

11. summary.txt
    High-level run summary:
    Total daily data points
    How many idle candidates
    How many rightsizing opportunities
    How many anomalies
    Forecast horizon
