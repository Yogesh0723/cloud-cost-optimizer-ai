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
