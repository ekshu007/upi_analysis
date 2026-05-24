"""
UPI Forecasting Script (Seasonal Naive)
--------------------------------------
✔ Uses best model (lag=7)
✔ Predicts future days
✔ Saves predictions
✔ Simple + production friendly

Run:
python predictor.py --days 14
"""

import os
import argparse
import pandas as pd
from datetime import timedelta

# ── Config ─────────────────────────────────────
DATA_PATH = "data/UPI_Master_2021_2026_Mar.csv"
OUTPUT_PATH = "data/predictions.csv"

# ── Load data ──────────────────────────────────
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Dataset not found")

    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

# ── Seasonal Naive Forecast ─────────────────────
def seasonal_naive_forecast(df, days=7, lag=7):
    forecasts = []

    last_date = df["Date"].max()

    for i in range(1, days + 1):
        target_date = last_date + timedelta(days=i)
        ref_date = target_date - timedelta(days=lag)

        # find value from last week
        ref_row = df[df["Date"] == ref_date]

        if not ref_row.empty:
            vol = ref_row["Volume (In Mn.)"].values[0]
            val = ref_row["Value (In Cr.)"].values[0]
        else:
            # fallback (if missing)
            vol = df["Volume (In Mn.)"].iloc[-lag]
            val = df["Value (In Cr.)"].iloc[-lag]

        forecasts.append({
            "Date": target_date.strftime("%Y-%m-%d"),
            "Pred_Volume (Mn)": round(vol, 2),
            "Pred_Value (Cr)": round(val, 2)
        })

    return pd.DataFrame(forecasts)

# ── Main ───────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()

    print("Loading data...")
    df = load_data()

    print(f"Predicting next {args.days} days...")
    pred_df = seasonal_naive_forecast(df, days=args.days)

    print("\nPredictions:")
    print(pred_df)

    pred_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH} ✅")

# ── Run ────────────────────────────────────────
if __name__ == "__main__":
    main()