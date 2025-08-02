import pandas as pd
act = pd.read_parquet("atus_analysis/data/processed/activity.parquet")
print(act[["ACT_CODE", "TUACTDUR24"]].head())