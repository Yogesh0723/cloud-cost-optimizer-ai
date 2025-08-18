import pandas as pd
import numpy as np
import random

# Generate mock GCP billing data for 5 months (~150 days)
dates = pd.date_range(start="2025-03-01", end="2025-07-31", freq="D")
projects = ["project-alpha", "project-beta"]
services = ["Compute Engine", "Cloud Storage", "BigQuery", "Cloud SQL", "Kubernetes Engine"]
skus = {
    "Compute Engine": ["n2-standard-4", "e2-standard-2", "n2-highmem-8"],
    "Cloud Storage": ["Standard Storage", "Nearline Storage", "Egress"],
    "BigQuery": ["Query Processing", "Active Storage"],
    "Cloud SQL": ["db-f1-micro", "db-n1-standard-2"],
    "Kubernetes Engine": ["GKE Nodes", "GKE Control Plane"]
}

rows = []
for date in dates:
    for project in projects:
        for service in services:
            sku = random.choice(skus[service])
            usage = round(random.uniform(1, 50), 2)  # usage amount
            cost = round(usage * random.uniform(0.1, 2.0), 2)  # random cost multiplier
            rows.append([project, service, sku, date, date, usage, cost])

# Create DataFrame
df = pd.DataFrame(rows, columns=[
    "project_id", "service", "sku_description",
    "usage_start_time", "usage_end_time",
    "usage_amount", "cost"
])

df.to_csv("gcp_billing_data.csv", index=False)

# Show preview
# import caas_jupyter_tools
# caas_jupyter_tools.display_dataframe_to_user("Mock Cloud Billing Data (5 months)", df)
