"""
FINAL VERSION — Stable Monthly UPI Updater
-----------------------------------------
✔ No Selenium (removed unreliable scraping)
✔ Uses NPCI monthly totals
✔ Generates realistic daily data
✔ Safe append to master CSV
✔ Ready for cron automation

Run:
python monthly_updater.py
python monthly_updater.py --year 2026 --month 3 --vol 12000 --val 1800000
"""

import os, argparse
import numpy as np
import pandas as pd
from datetime import date, datetime
from calendar import monthrange

# ── Config ─────────────────────────────────────────────
DATA_DIR    = "data"
MASTER_FILE = os.path.join(DATA_DIR, "UPI_Master_2021_2026_Mar.csv")

MONTH_NAMES = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
               7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}

# Minimal holidays (extend as needed)
HOLIDAYS = {
    "2026-01-26": "Republic Day",
    "2026-03-06": "Holi",
    "2026-08-15": "Independence Day",
    "2026-11-08": "Diwali",
    "2026-12-25": "Christmas"
}

# ── Utils ─────────────────────────────────────────────
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def get_previous_month():
    today = date.today()
    return (today.year - 1, 12) if today.month == 1 else (today.year, today.month - 1)

# ── Core: Monthly → Daily distribution ─────────────────
def distribute_month(year, month, vol, val):
    days = monthrange(year, month)[1]
    dates = pd.date_range(f"{year}-{month:02d}-01", periods=days)

    # smarter weights
    weights = []
    for d in dates:
        w = 1.0

        if d.weekday() >= 5:  # weekend dip
            w *= 0.85
        else:
            w *= 1.05

        if d.strftime("%Y-%m-%d") in HOLIDAYS:
            w *= 0.75

        weights.append(w)

    weights = np.array(weights)
    weights /= weights.sum()

    df = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Volume (In Mn.)": np.round(vol * weights, 2),
        "Value (In Cr.)": np.round(val * weights, 2),
    })

    return df

# ── Feature engineering ───────────────────────────────
def add_features(df, year, month):
    rows = []
    for _, r in df.iterrows():
        d = pd.to_datetime(r["Date"])
        ds = d.strftime("%Y-%m-%d")

        rows.append({
            "Date": ds,
            "Year": year,
            "Month": MONTH_NAMES[month],
            "Volume (In Mn.)": r["Volume (In Mn.)"],
            "Value (In Cr.)": r["Value (In Cr.)"],
            "Day_Name": d.strftime("%A"),
            "Day_Number": d.dayofweek,
            "Is_Weekend": int(d.dayofweek >= 5),
            "Is_Festival": int(ds in HOLIDAYS),
            "Festival_Name": HOLIDAYS.get(ds, "")
        })

    return pd.DataFrame(rows)

# ── Merge into master ─────────────────────────────────
def merge(new_df):
    if not os.path.exists(MASTER_FILE):
        raise FileNotFoundError("Master CSV not found")

    master = pd.read_csv(MASTER_FILE)

    last_date = pd.to_datetime(master["Date"]).max()
    new_df["Date"] = pd.to_datetime(new_df["Date"])

    new_rows = new_df[new_df["Date"] > last_date]

    if new_rows.empty:
        log("No new data needed")
        return

    new_rows["Date"] = new_rows["Date"].dt.strftime("%Y-%m-%d")

    combined = pd.concat([master, new_rows])
    combined.to_csv(MASTER_FILE, index=False)

    log(f"Added {len(new_rows)} rows")
    log(f"New range: {combined['Date'].iloc[0]} → {combined['Date'].iloc[-1]}")

# ── Main ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int)
    parser.add_argument("--month", type=int)
    parser.add_argument("--vol", type=float)
    parser.add_argument("--val", type=float)
    args = parser.parse_args()

    # Determine month
    if args.year and args.month:
        year, month = args.year, args.month
    else:
        year, month = get_previous_month()

    log(f"Updating {MONTH_NAMES[month]} {year}")

    # Get totals
    if args.vol and args.val:
        vol, val = args.vol, args.val
    else:
        print("\nEnter NPCI monthly totals:")
        vol = float(input("Volume (Mn): "))
        val = float(input("Value (Cr): "))

    # Generate daily data
    daily = distribute_month(year, month, vol, val)
    final = add_features(daily, year, month)

    print("\nPreview:")
    print(final.head())

    merge(final)

    log("Update complete ✅")

# ── Run ──────────────────────────────────────────────

if __name__ == "__main__":
    main()