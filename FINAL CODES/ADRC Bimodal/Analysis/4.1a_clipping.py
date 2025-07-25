import pandas as pd
from sklearn.linear_model import LinearRegression

# === Load original CSV ===
df = pd.read_csv("brainage_predictions_adrc_all_with_metadata.csv")

# === Ensure predicted age is numeric ===
df["Predicted_Age"] = pd.to_numeric(df["Predicted_Age"], errors="coerce")

# === Force clipping of values > 100 ===
clip_threshold = 120
df["Predicted_Age"] = df["Predicted_Age"].apply(lambda x: min(x, clip_threshold) if pd.notnull(x) else x)

# === Recalculate BAG and cBAG ===
df["BAG"] = df["Predicted_Age"] - df["Age"]
reg = LinearRegression().fit(df[["Age"]], df["BAG"])
df["cBAG"] = df["BAG"] - reg.predict(df[["Age"]])

# === Print sanity check ===
print(f"Max predicted age AFTER clipping: {df['Predicted_Age'].max():.2f}")

# === Save to fixed name ===
output_path = "brainage_predictions_adrc_all_clipped120.csv"
df.to_csv(output_path, index=False)
print(f"Saved: {output_path}")
