"""
app.py — UPI Forecasting Dashboard (NPCI Internal Tool)
============================================================
Run: streamlit run app.py

Features:
  • Data from master CSV (updated monthly via monthly_updater.py)
  • YoY same-day alerts (volume/value vs previous years same day)
  • Anomaly detection (z-score based)
  • Infrastructure risk calendar (festival/month-end scoring)
  • All model comparison charts
  • AI report generation via Ollama (local LLM)
"""
import google.generativeai as genai

import os, warnings, json, glob, requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
from calendar import monthrange
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="UPI NPCI Dashboard", page_icon="💳",
                   layout="wide", initial_sidebar_state="expanded")
# ── Seasonal Naive Forecast ─────────────────────────────────────────────
def seasonal_naive_forecast(series, steps=7, lag=7):
    """
    Forecast using Seasonal Naive:
    Next value = value from same season in past
    """
    forecast = []
    for i in range(steps):
        forecast.append(series.iloc[-(lag - i % lag)])
    return np.array(forecast)


# ── Forecast Alerts ─────────────────────────────────────────────────────
def compute_forecast_alerts(df, target, steps=7, lag=7):
    series = df[target]

    forecast = seasonal_naive_forecast(series, steps=steps, lag=lag)
    last_val = series.iloc[-1]

    alerts = []
    future_dates = [df["Date"].max() + pd.Timedelta(days=i+1) for i in range(steps)]

    for i, val in enumerate(forecast):
        change = (val - last_val) / last_val

        if abs(change) < 0.15:
            continue  # ignore small changes

        direction = "📈 Spike" if change > 0 else "📉 Drop"
        sev = ("🔴 CRITICAL" if abs(change) > 0.35 else
               "🟠 HIGH"     if abs(change) > 0.25 else
               "🟡 MEDIUM")

        alerts.append({
            "date": future_dates[i].strftime("%d %b %Y"),
            "day": future_dates[i].strftime("%A"),
            "forecast": round(val, 2),
            "change": round(change*100, 1),
            "severity": sev,
            "direction": direction
        })

    return forecast, alerts

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ---------- GLOBAL ---------- */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* 🔥 Proper dark background (new Streamlit fix) */
[data-testid="stAppViewContainer"] {
    background: #050B18;
}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0A1628,#0D2137);
    border-right: 1px solid #1E3A5F;
}
section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* ---------- TITLES ---------- */
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 32px;
    font-weight: 800;
    color: #FFFFFF;
    margin-bottom: 5px;
}

.page-sub {
    font-size: 14px;
    color: #A0AEC0;
}

.section-hdr {
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: #FFFFFF;
    margin: 20px 0 10px;
    border-bottom: 1px solid rgba(255,255,255,0.2);
}

/* ---------- ALERT BAR ---------- */
.alert-crit, .alert-high, .alert-med, .alert-low, .alert-info {
    background: #EDE0D4;
    border-left: 4px solid #FF8C00;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #000000;
    font-weight: 500;
}

/* ---------- METRIC CARDS ---------- */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #FF8C00, #FF6A00) !important;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 6px 18px rgba(255,140,0,0.25);
    color: white !important;
    overflow: visible !important;
}

/* Metric label */
div[data-testid="stMetric"] label {
    color: rgba(255,255,255,0.9) !important;
    font-size: 14px;
    white-space: normal !important;
}

/* Metric value */
div[data-testid="stMetric"] > div {
    color: #FFFFFF !important;
    font-weight: 700;
    font-size: 22px !important;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
}

/* ---------- REPORT BOX ---------- */
.report-box {
    background: #0F172A;
    border-radius: 12px;
    padding: 20px;
    border-top: 3px solid #FF8C00;
    color: #FFFFFF;
}

/* ---------- ABOUT / WHITE BOX FIX ---------- */
.white-box {
    background: #FFFFFF !important;
    color: #000000 !important;
}


/* ---------- SAFE TEXT RULE (NO GLOBAL FORCE) ---------- */
p, span {
    color: inherit;
}

</style>
""", unsafe_allow_html=True)
# ── Constants ─────────────────────────────────────────────────────────────────
# Base directory (where app.py is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Correct data path (works regardless of where Streamlit runs from)
DATA_PATH = os.path.join(BASE_DIR, "data", "UPI_Master_2021_2026_Mar.csv")

TRAIN_END   = "2025-09-30"
TEST_START  = "2025-10-01"

OLLAMA_URL  = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"   # change to whichever model you have pulled

COLORS = {"primary":"#1A6FBF","secondary":"#E07B39","success":"#2CA02C",
          "danger":"#D62728","purple":"#9467BD","dark":"#0A1628","gray":"#64748B"}

INDIAN_HOLIDAYS = {
    "2026-01-14":("Makar Sankranti","Public"),  "2026-01-26":("Republic Day","National"),
    "2026-02-26":("Maha Shivaratri","Public"),   "2026-03-06":("Holi","Public"),
    "2026-03-25":("Gudi Padwa","Public"),         "2026-03-30":("Ram Navami","Public"),
    "2026-04-03":("Good Friday","Public"),         "2026-04-14":("Ambedkar Jayanti","Public"),
    "2026-05-01":("Maharashtra Day","Public"),     "2026-05-23":("Buddha Purnima","Public"),
    "2026-06-19":("Eid ul-Adha","Public"),         "2026-08-15":("Independence Day","National"),
    "2026-08-23":("Janmashtami","Public"),          "2026-10-02":("Gandhi Jayanti","National"),
    "2026-10-22":("Dussehra","Public"),             "2026-11-08":("Diwali","Public"),
    "2026-11-24":("Guru Nanak Jayanti","Public"),  "2026-12-25":("Christmas Day","Public"),
}

# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["Festival_Name"] = df["Festival_Name"].fillna("")
    df["Holiday_Type"]  = df["Holiday_Type"].fillna("Unknown")
    df["Month_Num"]     = df["Date"].dt.month
    df["Quarter"]       = df["Date"].dt.quarter
    df["YearStr"]       = df["Date"].dt.year.astype(str)
    df["MonthName"]     = df["Date"].dt.strftime("%b")
    df["DayOfYear"]     = df["Date"].dt.dayofyear
    return df

@st.cache_data
def load_metrics():
    records = []
    for path in glob.glob(os.path.join(BASE_DIR, "models", "*_metrics.json")):
        with open(path) as f:
            try: records.append(json.load(f))
            except: pass
    return pd.DataFrame(records) if records else pd.DataFrame()

df    = load_data()
train = df[df["Date"] <= TRAIN_END].copy()
test  = df[df["Date"] >= TEST_START].copy()


# ── Alert Engine ──────────────────────────────────────────────────────────────
def compute_yoy_alerts(df, target, n_days=30):
    """
    Smart YoY alert — only compares to PREVIOUS YEAR (offset=1).
    Flags only when deviation from the EXPECTED growth trend is significant.
    
    Why: UPI grows ~40% YoY. Comparing 2026 to 2023 gives 180% which is 
    normal growth not an alert. We compute the expected growth rate from 
    recent year-pairs and only alert when actual deviates from that expectation.
    """
    alerts = []
    recent = df.tail(n_days).copy()
    real_data_end = "2025-12-31"
    real_df = df[df["Date"] <= real_data_end].copy()
    recent  = real_df.tail(n_days).copy()

    # Step 1: Compute expected YoY growth from all available year-pairs
    yearly_avg = df.groupby("YearStr")[target].mean()
    years_avail = sorted(yearly_avg.index.tolist())
    pairs = []
    for i in range(len(years_avail) - 1):
        y0, y1 = years_avail[i], years_avail[i+1]
        if yearly_avg[y0] > 0:
            pairs.append((yearly_avg[y1] - yearly_avg[y0]) / yearly_avg[y0] * 100)
    # Use median — robust to outliers like COVID dip year
    expected_yoy_pct = round(np.median(pairs), 1) if pairs else 40.0

    for _, row in recent.iterrows():
        d   = row["Date"]
        val = float(row[target])
        if pd.isna(val): continue

        # Step 2: ONLY compare to 1 year ago — not 2 or 3
        prev_date = d - pd.DateOffset(years=1)
        match = df[df["Date"].dt.date == prev_date.date()]
        if match.empty:
            # Fallback: same weekday in same month of previous year
            match = df[
                (df["Date"].dt.year  == prev_date.year) &
                (df["Date"].dt.month == prev_date.month) &
                (df["Date"].dt.dayofweek == d.dayofweek)
            ]
            if match.empty:
                continue
            match = match.iloc[0:1]
            match_type = "same_weekday"
        else:
            match_type = "exact"

        prev_val = float(match[target].values[0])
        if prev_val == 0:
            continue

        # Step 3: Raw % change vs last year
        raw_pct = (val - prev_val) / prev_val * 100

        # Step 4: Deviation from EXPECTED growth
        # e.g. if expected is 40% and actual is 43% → deviation = +3% → no alert
        # if expected is 40% and actual is 15%  → deviation = -25% → HIGH alert
        deviation = raw_pct - expected_yoy_pct

        # Step 5: Only alert if deviation from trend is significant (>15%)
        if abs(deviation) < 15:
            continue

        direction = "📈 ABOVE TREND" if deviation > 0 else "📉 BELOW TREND"
        context   = (f"+{deviation:.1f}% above expected {expected_yoy_pct:.1f}% growth"
                     if deviation > 0 else
                     f"{deviation:.1f}% below expected {expected_yoy_pct:.1f}% growth")

        # Severity based on deviation from trend (not raw %)
        abs_dev = abs(deviation)
        sev = ("🔴 CRITICAL" if abs_dev > 50 else
               "🟠 HIGH"     if abs_dev > 30 else
               "🟡 MEDIUM"   if abs_dev > 15 else
               "🟢 LOW")

        alerts.append({
            "date":      d.strftime("%d %b %Y"),
            "day":       d.strftime("%A"),
            "current":   round(val, 2),
            "prev_year": d.year - 1,
            "prev_val":  round(prev_val, 2),
            "raw_pct":   round(raw_pct, 1),
            "expected":  round(expected_yoy_pct, 1),
            "deviation": round(deviation, 1),
            "severity":  sev,
            "match":     match_type,
            "direction": direction,
            "context":   context
        })

    return sorted(alerts, key=lambda x: abs(x["deviation"]), reverse=True)


def compute_anomaly_alerts(df, target, n_days=30, window=30):
    """Z-score anomaly detection."""
    alerts = []
    for _, row in df.tail(n_days).iterrows():
        d, val = row["Date"], row[target]
        if pd.isna(val): continue
        recent = df[df["Date"] < d][target].tail(window)
        if len(recent) < 7: continue
        mu, sigma = recent.mean(), recent.std()
        if sigma == 0: continue
        z = (val - mu) / sigma
        if abs(z) <= 2.0: continue
        sev = ("🔴 CRITICAL" if abs(z)>3.5 else
               "🟠 HIGH"     if abs(z)>3.0 else
               "🟡 MEDIUM"   if abs(z)>2.5 else "🟢 LOW")
        alerts.append({
            "date": d.strftime("%d %b %Y"), "value": round(val, 2),
            "mean": round(mu, 2), "z": round(z, 3), "severity": sev,
            "dir": "⬆️ spike" if z > 0 else "⬇️ dip",
        })
    return sorted(alerts, key=lambda x: abs(x["z"]), reverse=True)


def compute_risk_calendar(n_days=14):
    """Score upcoming days for infrastructure risk."""
    hol_pds = pd.to_datetime(list(INDIAN_HOLIDAYS.keys()))
    last    = df["Date"].max()
    days    = []
    for i in range(1, n_days + 1):
        fd   = last + pd.Timedelta(days=i)
        ds   = fd.strftime("%Y-%m-%d")
        dow  = fd.dayofweek
        score, flags = 0, []

        if ds in INDIAN_HOLIDAYS:
            score += 30; flags.append(f"🎉 {INDIAN_HOLIDAYS[ds][0]}")
        prev = (fd - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        nxt  = (fd + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if prev in INDIAN_HOLIDAYS or nxt in INDIAN_HOLIDAYS:
            score += 15; flags.append("📅 Adjacent to holiday")
        if fd.day >= 28:
            score += 25; flags.append(f"📆 Month-end (Day {fd.day})")
        if dow == 0:
            score += 10; flags.append("📈 Monday catchup")
        if dow >= 5 and any(abs((fd-h).days) <= 2 for h in hol_pds):
            score += 20; flags.append("🏖️ Long weekend")
        cluster = sum(1 for h in hol_pds if abs((fd-h).days) <= 3)
        if cluster >= 2:
            score += cluster * 8; flags.append(f"🗓️ {cluster} holidays nearby")

        # Historical context
        py = df[df["Date"].dt.date == (fd - pd.DateOffset(years=1)).date()]
        if not py.empty:
            pv = py["Volume (In Mn.)"].values[0]
            om = df["Volume (In Mn.)"].mean()
            if pv > om * 1.15:
                score += 15; flags.append(f"📊 Last year: {pv:.0f} Mn (+{(pv/om-1)*100:.0f}% vs avg)")

        if score == 0: continue
        sev = ("🔴 CRITICAL" if score>=60 else "🟠 HIGH" if score>=40 else
               "🟡 MEDIUM"   if score>=25 else "🟢 LOW")
        days.append({"date": fd.strftime("%d %b %Y"), "date_raw": fd,
                     "day": fd.strftime("%A"), "severity": sev,
                     "score": score, "flags": flags,
                     "is_fest": ds in INDIAN_HOLIDAYS,
                     "fest_name": INDIAN_HOLIDAYS.get(ds, ("",""))[0]})

    return sorted(days, key=lambda x: (-x["score"], x["date_raw"]))


def alert_card(a, css_class, body):
    st.markdown(f'<div class="{css_class}">{body}</div>', unsafe_allow_html=True)


def sev_to_css(sev):
    return ("alert-crit" if "CRITICAL" in sev else "alert-high" if "HIGH" in sev else
            "alert-med"  if "MEDIUM"   in sev else "alert-low")


# ── Ollama Report Generator ───────────────────────────────────────────────────
def generate_report_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Call local Ollama API to generate a text report."""
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 800},
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        if resp.status_code == 200:
            return resp.json().get("response", "No response from model.")
        return f"Ollama returned HTTP {resp.status_code}. Make sure Ollama is running (`ollama serve`)."
    except requests.exceptions.ConnectionError:
        return ("❌ Cannot connect to Ollama. Make sure it is running:\n"
                "  1. Install: https://ollama.com\n"
                "  2. Run: ollama serve\n"
                "  3. Pull model: ollama pull llama3.2\n"
                "  4. Restart this dashboard.")
    except Exception as e:
        return f"Error: {e}"


def build_report_prompt(df, target, n_days=30, risk_days=None, yoy_alerts=None):
    """Build a structured prompt for the Ollama report."""
    recent   = df.tail(n_days)
    last_val = recent[target].iloc[-1]
    avg_30   = recent[target].mean()
    max_30   = recent[target].max()
    min_30   = recent[target].min()
    trend    = "increasing" if recent[target].iloc[-7:].mean() > recent[target].iloc[-14:-7].mean() else "decreasing"

    # YoY comparison
    last_date  = df["Date"].max()
    prev_yr    = df[df["Date"].dt.date == (last_date - pd.DateOffset(years=1)).date()]
    prev_val   = prev_yr[target].values[0] if not prev_yr.empty else None
    yoy_pct    = round((last_val - prev_val) / prev_val * 100, 1) if prev_val else "N/A"

    # Top alerts summary
    top_alerts = ""
    if yoy_alerts:
        for a in yoy_alerts[:3]:
            top_alerts += f"  - {a['date']}: {a['raw_pct']:+.1f}% vs {a['prev_year']} ({a['direction']})\n"

    # Risk days summary
    risk_summary = ""
    if risk_days:
        for r in risk_days[:3]:
            risk_summary += f"  - {r['date']} ({r['day']}): {r['severity']} — {', '.join(r['flags'][:2])}\n"

    prompt = f"""You are an analyst at NPCI (National Payments Corporation of India) preparing a 
daily UPI transaction monitoring report. Write a concise, professional report in 4 sections:

1. CURRENT STATUS — 2-3 sentences on today's {target} level and recent trend
2. YEAR-OVER-YEAR ANALYSIS — 2-3 sentences on YoY performance
3. UPCOMING RISK DAYS — 2-3 sentences on infrastructure planning for upcoming high-risk days
4. RECOMMENDATION — 1-2 clear action items for NPCI infrastructure team

DATA SUMMARY:
- Latest {target}: {last_val:.2f}
- 30-day average: {avg_30:.2f}
- 30-day max: {max_30:.2f}, min: {min_30:.2f}
- 7-day trend: {trend}
- YoY change (vs same day last year): {yoy_pct}%
- Data through: {last_date.strftime('%d %B %Y')}

RECENT YoY ALERTS:
{top_alerts if top_alerts else '  None significant'}

UPCOMING HIGH-RISK DAYS:
{risk_summary if risk_summary else '  None in next 14 days'}

Write the report now. Be specific with numbers. Keep it under 300 words. Do not use bullet points inside sections.
"""
    return prompt


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:14px 0 6px;'>
        <div style='font-family:Syne,sans-serif;font-size:19px;font-weight:800;color:white;'>
            💳 UPI NPCI Dashboard
        </div>
        <div style='font-size:11px;color:#64748B;margin-top:3px;'>
            <span class="live-dot"></span>Live Monitoring System
        </div>
    </div>
    <hr style='border-color:#1E3A5F;margin:8px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Overview",
        "🚨  Alert Centre",
        "📊  YoY Analysis",
        "🔮  Risk Calendar",
        "📈  Trend Analysis",
        "🔬  Statistical Analysis",
        "🏆  Model Comparison",
        "🤖  AI Report"
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1E3A5F;margin:8px 0;'>", unsafe_allow_html=True)
    target = st.selectbox("🎯 Target", ["Volume (In Mn.)", "Value (In Cr.)"])

    st.markdown(f"""
    <div style='font-size:11px;color:#64748B;padding:6px 0;'>
        📅 Data: May 2021 – Mar 2026<br>
        📋 Records: {len(df):,} days<br>
        🕐 Last updated: {datetime.now().strftime('%d %b %Y %H:%M')}
    </div>""", unsafe_allow_html=True)

    if st.button("🔄 Reload Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown('<div class="page-title">💳 UPI Overview</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub"><span class="live-dot"></span>Real-time monitoring · Data through {df["Date"].max().strftime("%d %b %Y")}</div>', unsafe_allow_html=True)

    # ── Risk banner ───────────────────────────────────────────────────────────
    risk_days = compute_risk_calendar(14)
    critical  = [r for r in risk_days if "CRITICAL" in r["severity"]]
    high      = [r for r in risk_days if "HIGH"     in r["severity"]]
    next_risk = risk_days[0] if risk_days else None

    overall_sev = ("🔴 CRITICAL" if critical else "🟠 HIGH" if high else
                   "🟡 MEDIUM"   if risk_days else "🟢 LOW")
    css_banner  = sev_to_css(overall_sev)
    next_txt    = (f"Next: <strong>{next_risk['date']}</strong> ({next_risk['severity']}) — "
                   f"{', '.join(next_risk['flags'][:2])}" if next_risk else "No high-risk days in next 14 days")
    st.markdown(f'<div class="{css_banner}"><strong>🏗️ Infrastructure Risk: {overall_sev}</strong> &nbsp;|&nbsp; {len(critical)} critical, {len(high)} high risk days ahead &nbsp;|&nbsp; {next_txt}</div>',
                unsafe_allow_html=True)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    latest_vol = df["Volume (In Mn.)"].iloc[-1]
    latest_val = df["Value (In Cr.)"].iloc[-1]
    avg30_vol  = df["Volume (In Mn.)"].tail(30).mean()
    prev1y     = df[df["Date"].dt.date == (df["Date"].max() - pd.DateOffset(years=1)).date()]
    yoy_vol    = f"{(latest_vol/prev1y['Volume (In Mn.)'].values[0]-1)*100:+.1f}% YoY" if not prev1y.empty else ""
    yoy_val    = f"{(latest_val/prev1y['Value (In Cr.)'].values[0]-1)*100:+.1f}% YoY"  if not prev1y.empty else ""
    yoy_       = df.groupby("YearStr")["Volume (In Mn.)"].sum()
    cagr_      = ((yoy_.get("2024",1)/yoy_.get("2022",1))**0.5-1)*100

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.metric("📅 Data Through",    df["Date"].max().strftime("%d %b %Y"))
    with c2: st.metric("📦 Latest Volume",   f"{latest_vol:.0f} Mn",  delta=yoy_vol)
    with c3: st.metric("💰 Latest Value",    f"₹{latest_val:,.0f} Cr",delta=yoy_val)
    with c4: st.metric("📊 30d Avg Volume",  f"{avg30_vol:.1f} Mn")
    with c5: st.metric("📈 CAGR 2022–24",   f"{cagr_:.1f}%")

    st.markdown("---")

    # ── Main trend chart ──────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Full Daily Trend (May 2021 – Mar 2026)</div>', unsafe_allow_html=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Volume (Mn.)", "Value (₹ Cr.)"], vertical_spacing=0.07)
    for r, col, c in [(1,"Volume (In Mn.)",COLORS["primary"]),(2,"Value (In Cr.)",COLORS["secondary"])]:
        fig.add_trace(go.Scatter(x=df["Date"], y=df[col], mode="lines", name=col,
                                  line=dict(color=c, width=0.8), opacity=0.5), row=r, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df[col].rolling(30).mean(), mode="lines",
                                  name="30d MA", line=dict(color=COLORS["danger"], width=2.5)), row=r, col=1)
    # Mark festivals
    for fd in df[df["Is_Festival"]==1]["Date"].tolist()[:30]:
        fig.add_vline(x=fd, line_dash="dot", line_color="rgba(148,100,200,0.25)", line_width=1)
    fig.update_layout(height=460, template="plotly_white", hovermode="x unified", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Dotted lines = Indian public holidays")

    # ── Recent 30 days ────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Last 30 Days Detail</div>', unsafe_allow_html=True)
    r30 = df.tail(30).copy()
    bar_colors = [COLORS["secondary"] if w else COLORS["primary"] for w in r30["Is_Weekend"]]
    festival_markers = [("⭐ "+fn if fn else "") for fn in r30["Festival_Name"]]

    fig_r30 = go.Figure()
    fig_r30.add_trace(go.Bar(x=r30["Date"], y=r30[target], marker_color=bar_colors,
                              name=target, text=festival_markers,
                              hovertemplate="%{x}<br>%{y:.1f}<br>%{text}"))
    fig_r30.add_trace(go.Scatter(x=r30["Date"], y=r30[target].rolling(7,min_periods=1).mean(),
                                  mode="lines", name="7d MA",
                                  line=dict(color=COLORS["danger"], width=2.5)))
    fig_r30.update_layout(template="plotly_white", height=300, bargap=0.2,
                           title=f"Last 30 Days — {target} (Blue=Weekday | Orange=Weekend)",
                           hovermode="x unified")
    st.plotly_chart(fig_r30, use_container_width=True)

    # ── Key insights ──────────────────────────────────────────────────────────
    c1,c2,c3 = st.columns(3)
    growth = (df["Volume (In Mn.)"].iloc[-1] / df["Volume (In Mn.)"].iloc[0] - 1) * 100
    sun_avg = df[df["Day_Name"]=="Sunday"]["Volume (In Mn.)"].mean()
    mon_avg = df[df["Day_Name"]=="Monday"]["Volume (In Mn.)"].mean()
    drop    = (mon_avg - sun_avg) / mon_avg * 100
    fest_avg = df[df["Is_Festival"]==1]["Volume (In Mn.)"].mean()
    norm_avg = df[df["Is_Festival"]==0]["Volume (In Mn.)"].mean()
    fest_drop= (norm_avg - fest_avg) / norm_avg * 100
    with c1: st.markdown(f'<div class="alert-info"><strong>📈 {growth:.0f}% Growth:</strong> May 2021 → Mar 2026</div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="alert-info"><strong>📅 Sunday Dip:</strong> {drop:.1f}% below Monday average</div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="alert-info"><strong>🎉 Festival Effect:</strong> {fest_drop:.1f}% below non-festival days</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ALERT CENTRE
# ══════════════════════════════════════════════════════════════════════════════
elif "Alert" in page:
    st.markdown('<div class="page-title">🚨 Alert Centre</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">YoY deviations · anomaly detection · infrastructure risk</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 YoY Alerts",
        "🔍 Anomaly Detection",
        "🏗️ Infrastructure",
        "🔮 Forecast Alerts"
    ])

    # ───────────────── TAB 1 ─────────────────
    with tab1:
        n_days = st.slider("Look back (days)", 7, 60, 30, key="yoy_n")
        alerts = compute_yoy_alerts(df, target, n_days)

        if not alerts:
            st.markdown('<div class="alert-low">✅ No significant YoY deviations.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"**{len(alerts)} alert(s):**")
            for a in alerts[:20]:
                css = sev_to_css(a["severity"])
                st.markdown(f"""
                <div class="{css}">
                <strong>{a['severity']} — {a['date']} ({a['day']})</strong><br>
                Current: <strong>{a['current']:.1f}</strong> |
                Prev: <strong>{a['prev_val']:.1f}</strong> |
                Change: <strong>{a.get('raw_pct'):+.1f}%</strong>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="section-hdr">Current vs Previous Year</div>', unsafe_allow_html=True)

        recent = df.tail(n_days)
        prev_y = df[df["YearStr"] == str(df["Date"].max().year - 1)].tail(n_days)

        fig_yoy = go.Figure()
        fig_yoy.add_trace(go.Scatter(
            x=recent["Date"], y=recent[target],
            mode="lines+markers", name="Current"
        ))

        if len(prev_y) > 0:
            fig_yoy.add_trace(go.Scatter(
                x=recent["Date"].values,
                y=prev_y[target].values[:len(recent)],
                mode="lines", name="Prev Year",
                line=dict(dash="dash")
            ))

        st.plotly_chart(fig_yoy, use_container_width=True)

    # ───────────────── TAB 2 ─────────────────
    with tab2:
        n_anom = st.slider("Look back (days)", 7, 60, 30, key="anom_n")
        anom_alerts = compute_anomaly_alerts(df, target, n_anom)

        if not anom_alerts:
            st.markdown('<div class="alert-low">✅ No anomalies detected.</div>', unsafe_allow_html=True)
        else:
            for a in anom_alerts:
                css = sev_to_css(a["severity"])
                st.markdown(f"""
                <div class="{css}">
                <strong>{a['severity']} — {a['date']}</strong> {a['dir']}<br>
                Actual: <strong>{a['value']:.1f}</strong> |
                Mean: <strong>{a['mean']:.1f}</strong> |
                Z-score: <strong>{a['z']:+.2f}</strong>
                </div>
                """, unsafe_allow_html=True)

    # ───────────────── TAB 3 ─────────────────
    with tab3:
        n_fwd = st.slider("Days ahead", 7, 30, 14, key="risk_n")
        risk_days = compute_risk_calendar(n_fwd)

        for rd in risk_days:
            css = sev_to_css(rd["severity"])
            st.markdown(f"""
            <div class="{css}">
            <strong>{rd['severity']} — {rd['date']} ({rd['day']})</strong>
            </div>
            """, unsafe_allow_html=True)

    # ───────────────── TAB 4 ─────────────────
    with tab4:
        st.markdown("### 🔮 Future Forecast Alerts (Seasonal Naive Model)")

        steps = st.slider("Forecast Days Ahead", 3, 14, 7)
        lag = 7

        forecast, f_alerts = compute_forecast_alerts(df, target, steps, lag)

        if not f_alerts:
            st.markdown('<div class="alert-low">✅ No significant future risks detected.</div>', unsafe_allow_html=True)
        else:
            for a in f_alerts:
                css = sev_to_css(a["severity"])
                st.markdown(f"""
                <div class="{css}">
                <strong>{a['severity']} — {a['date']} ({a['day']})</strong><br>
                Forecast: <strong>{a['forecast']}</strong><br>
                Change: <strong>{a['change']:+.1f}%</strong> {a['direction']}
                </div>
                """, unsafe_allow_html=True)

        future_dates = [df["Date"].max() + pd.Timedelta(days=i+1) for i in range(steps)]

        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(
            x=df["Date"].tail(30),
            y=df[target].tail(30),
            mode="lines",
            name="Actual"
        ))

        fig_f.add_trace(go.Scatter(
            x=future_dates,
            y=forecast,
            mode="lines+markers",
            name="Forecast",
            line=dict(dash="dash")
        ))

        fig_f.update_layout(template="plotly_white", height=350, title="Actual vs Forecast")
        st.plotly_chart(fig_f, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — YoY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif "YoY" in page:
    st.markdown('<div class="page-title">📊 Year-over-Year Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Compare any date across all available years</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📅 Same-Day Comparison", "📆 Monthly YoY Heatmap"])

    # ── TAB 1: SAME DAY COMPARISON ─────────────────────────
    with tab1:
        sel = st.date_input(
            "Select date",
            value=df["Date"].max().date(),
            min_value=df["Date"].min().date(),
            max_value=df["Date"].max().date()
        )

        sel_ts = pd.Timestamp(sel)
        curr = df[df["Date"].dt.date == sel]
        curr_val = curr[target].values[0] if not curr.empty else None

        if curr_val:
            rows = [{
                "Year": str(sel_ts.year),
                "Date": str(sel),
                "Value": curr_val,
                "YoY%": "—"
            }]

            for off in [1, 2, 3, 4]:
                prev_d = sel_ts - pd.DateOffset(years=off)
                prow = df[df["Date"].dt.date == prev_d.date()]
                if prow.empty:
                    continue

                pv = prow[target].values[0]

                rows.append({
                    "Year": str(sel_ts.year - off),
                    "Date": str(prev_d.date()),
                    "Value": round(pv, 2),
                    "YoY%": f"{(curr_val - pv) / pv * 100:+.1f}%"
                })

            comp_df = pd.DataFrame(rows)

            fig_c = px.bar(
                comp_df,
                x="Year",
                y="Value",
                text="Value",
                color="Year",
                color_discrete_sequence=[
                    COLORS["primary"],
                    COLORS["secondary"],
                    COLORS["success"],
                    COLORS["danger"],
                    COLORS["purple"]
                ]
            )

            fig_c.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_c.update_layout(
                template="plotly_white",
                height=350,
                showlegend=False,
                title=f"{target} on {sel_ts.strftime('%d %b')} across years"
            )

            st.plotly_chart(fig_c, use_container_width=True)
            st.dataframe(comp_df, use_container_width=True)

    # ── TAB 2: MONTHLY HEATMAP ─────────────────────────────
    with tab2:
        pivot = df.groupby(["YearStr", "Month_Num"])[target].mean().unstack()

        mn = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot.columns = mn[:len(pivot.columns)]

        fig_hm = px.imshow(
            pivot,
            text_auto=".0f",
            color_continuous_scale="YlOrRd",
            title=f"Avg Daily {target} by Month × Year"
        )
        fig_hm.update_layout(template="plotly_white", height=450)
        st.plotly_chart(fig_hm, use_container_width=True)

        pct = pivot.pct_change() * 100

        fig_pct = px.imshow(
            pct.round(1),
            text_auto=".1f",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            title="YoY % Change by Month"
        )
        fig_pct.update_layout(template="plotly_white", height=450)
        st.plotly_chart(fig_pct, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RISK CALENDAR
# ══════════════════════════════════════════════════════════════════════════════
elif "Calendar" in page:
    st.markdown('<div class="page-title">🔮 Infrastructure Risk Calendar</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Upcoming high-risk days based on festivals, month-end and historical patterns</div>', unsafe_allow_html=True)

    n_fwd = st.slider("Days ahead", 7, 60, 30)
    risk_days = compute_risk_calendar(n_fwd)

    if not risk_days:
        st.markdown('<div class="alert-low">✅ No risk days detected.</div>', unsafe_allow_html=True)
    else:
        rdf = pd.DataFrame(risk_days)
        color_map = {"🔴 CRITICAL":COLORS["danger"],"🟠 HIGH":COLORS["secondary"],
                      "🟡 MEDIUM":"#D4A017","🟢 LOW":COLORS["success"]}
        fig_cal = px.bar(rdf, x="date", y="score", color="severity",
                          color_discrete_map=color_map, text="score",
                          title=f"Infrastructure Risk — Next {n_fwd} Days",
                          hover_data={"date":True,"severity":True,"score":True})
        fig_cal.update_traces(textposition="outside")
        fig_cal.update_layout(template="plotly_white", height=380, xaxis_tickangle=30)
        st.plotly_chart(fig_cal, use_container_width=True)

        st.markdown('<div class="section-hdr">Detailed Risk Breakdown + Last Year Reference</div>', unsafe_allow_html=True)
        for rd in risk_days:
            css = sev_to_css(rd["severity"])
            flags_html = "".join([f'<span style="background:#EEE;border-radius:4px;padding:2px 6px;margin:2px;font-size:12px;">{f}</span>' for f in rd["flags"]])
            py_row = df[df["Date"].dt.date == (rd["date_raw"] - pd.DateOffset(years=1)).date()]
            hist = f"<br><span style='font-size:12px;color:#1A6FBF;'>📊 Last year this day: <strong>{py_row['Volume (In Mn.)'].values[0]:.0f} Mn transactions</strong></span>" if not py_row.empty else ""
            st.markdown(f"""<div class="{css}">
            <strong>{rd['severity']}</strong> &nbsp; <span style='font-size:16px;font-weight:700;'>{rd['date']}</span>
            &nbsp; ({rd['day']}) &nbsp; <span style='font-size:11px;color:#888;'>Score: {rd['score']}</span><br>
            <div style='margin-top:5px;'>{flags_html}</div>{hist}
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — TREND ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif "Trend" in page:
    st.markdown('<div class="page-title">📈 Trend Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Rolling Stats","🗓️ Weekly Patterns","📆 Monthly/Quarterly"])

    with tab1:
        rw = st.slider("Window",7,90,30)
        rm = df[target].rolling(rw).mean()
        rs = df[target].rolling(rw).std()
        fig_r = make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.07)
        fig_r.add_trace(go.Scatter(x=df["Date"],y=df[target],mode="lines",opacity=0.3,
                                    line=dict(color=COLORS["primary"],width=0.8),name="Daily"),row=1,col=1)
        fig_r.add_trace(go.Scatter(x=df["Date"],y=rm,mode="lines",
                                    line=dict(color=COLORS["danger"],width=2.5),name=f"{rw}d MA"),row=1,col=1)
        fig_r.add_trace(go.Scatter(x=df["Date"],y=rs,mode="lines",
                                    line=dict(color=COLORS["purple"],width=1.5),
                                    fill="tozeroy",fillcolor="rgba(148,103,189,0.1)",name="Volatility"),row=2,col=1)
        fig_r.update_layout(height=480,template="plotly_white",hovermode="x unified")
        st.plotly_chart(fig_r,use_container_width=True)

    with tab2:
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        day_avg   = df.groupby("Day_Name")[target].mean().reindex(day_order)
        overall   = df[target].mean()
        pct       = ((day_avg-overall)/overall*100).round(1)
        colors_   = [COLORS["primary"]]*5+[COLORS["secondary"]]*2
        fig_dw = go.Figure(go.Bar(x=day_order,y=day_avg,marker_color=colors_,
                                   text=[f"{p:+.1f}%" for p in pct],textposition="outside"))
        fig_dw.add_hline(y=overall,line_dash="dash",line_color="black",
                          annotation_text=f"Overall avg: {overall:.1f}")
        fig_dw.update_layout(template="plotly_white",height=350,title=f"Avg {target} by Day of Week")
        st.plotly_chart(fig_dw,use_container_width=True)
        fig_box = px.box(df,x="Day_Name",y=target,category_orders={"Day_Name":day_order},
                          color="Day_Name",color_discrete_sequence=colors_,title="Distribution by Day")
        fig_box.update_layout(template="plotly_white",height=320,showlegend=False)
        st.plotly_chart(fig_box,use_container_width=True)

    with tab3:
        mn_names=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        m_avg = df.groupby("Month_Num")[target].mean().reindex(range(1,13))
        m_avg.index = mn_names
        fig_m = px.bar(x=mn_names,y=m_avg.values,color=m_avg.values,color_continuous_scale="YlOrRd",
                        text=[f"{v:.0f}" for v in m_avg.values],title=f"Monthly Avg {target}")
        fig_m.update_layout(template="plotly_white",height=320,showlegend=False)
        st.plotly_chart(fig_m,use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif "Statistical" in page:
    st.markdown('<div class="page-title">🔬 Statistical Analysis</div>', unsafe_allow_html=True)
    try:
        from statsmodels.tsa.seasonal import STL
        from statsmodels.tsa.stattools import adfuller
        import matplotlib.pyplot as plt
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

        series = df.set_index("Date")[target]
        tab1, tab2, tab3 = st.tabs(["🧪 Stationarity","🔀 Decomposition","📉 ACF/PACF"])

        with tab1:
            adf_r = adfuller(series.dropna(), autolag="AIC")
            adf_d = adfuller(series.diff().dropna(), autolag="AIC")
            c1,c2 = st.columns(2)
            with c1:
                st.markdown("**Raw Series**")
                s1 = "✅ Stationary" if adf_r[1]<0.05 else "❌ Non-Stationary"
                st.metric("ADF Stat", f"{adf_r[0]:.4f}")
                st.metric("p-value",  f"{adf_r[1]:.6f}", delta=s1)
            with c2:
                st.markdown("**1st Differenced**")
                s2 = "✅ Stationary" if adf_d[1]<0.05 else "❌ Non-Stationary"
                st.metric("ADF Stat", f"{adf_d[0]:.4f}")
                st.metric("p-value",  f"{adf_d[1]:.6f}", delta=s2)

        with tab2:
            stl = STL(series, period=7, robust=True).fit()
            fig_stl = make_subplots(rows=4,cols=1,shared_xaxes=True,vertical_spacing=0.04,
                                     subplot_titles=["Observed","Trend","Seasonal","Residual"])
            for i,(comp,col) in enumerate(zip([series,stl.trend,stl.seasonal,stl.resid],
                                               [COLORS["primary"],COLORS["danger"],COLORS["secondary"],"gray"]),1):
                fig_stl.add_trace(go.Scatter(x=comp.index,y=comp.values,mode="lines",
                                              line=dict(color=col,width=1),showlegend=False),row=i,col=1)
            fig_stl.update_layout(height=600,template="plotly_white",title=f"STL — {target}")
            st.plotly_chart(fig_stl,use_container_width=True)

        with tab3:
            diff1 = series.diff().dropna()
            fig_acf, axes = plt.subplots(2,2,figsize=(14,7))
            plot_acf (series.dropna(),ax=axes[0][0],lags=50,title="ACF — Raw")
            plot_pacf(series.dropna(),ax=axes[0][1],lags=50,title="PACF — Raw",method="ywm")
            plot_acf (diff1.dropna(), ax=axes[1][0],lags=50,title="ACF — 1st Diff")
            plot_pacf(diff1.dropna(), ax=axes[1][1],lags=50,title="PACF — 1st Diff",method="ywm")
            for ax in axes.flat: ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_acf)
            plt.close()
    except ImportError:
        st.warning("statsmodels not installed: pip install statsmodels")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif "Comparison" in page:
    st.markdown('<div class="page-title">🏆 Model Comparison</div>', unsafe_allow_html=True)
    metrics_df = load_metrics()

    if metrics_df.empty:
        st.markdown('<div class="alert-high"><strong>⚠️ No metrics found.</strong> Run all model notebooks first.</div>', unsafe_allow_html=True)
    else:
        def tag(n):
            nl=str(n).lower()
            if any(x in nl for x in ["mean","median","drift","naive","rolling"]): return "Baseline"
            if any(x in nl for x in ["ses","holt","winters","ets"]): return "Exp. Smoothing"
            if any(x in nl for x in ["mstl","bats","tbats"]): return "Multi-Seasonal"
            if "sarimax" in nl or "sarima" in nl: return "Statistical"
            if "arima" in nl: return "Statistical"
            if "prophet" in nl: return "Statistical/ML"
            if "lstm" in nl: return "Deep Learning"
            return "Other"

        metrics_df["type"] = metrics_df["model"].apply(tag)
        tf  = "Volume" if "Volume" in target else "Value"
        sub = metrics_df[metrics_df["target"].str.contains(tf,case=False)].sort_values("MAPE").reset_index(drop=True)
        sub.index += 1

        best = sub.iloc[0]
        base_df = sub[sub["type"] == "Baseline"]
        if not base_df.empty:
            bbase = base_df["MAPE"].min()
            # Avoid division issues
            if bbase > 0:
                imp = (bbase - best["MAPE"]) / bbase * 100
                msg = f'<strong>{imp:.1f}% improvement over best baseline</strong>'
            else:
                msg = "Baseline comparison unavailable"
        else:
            msg = "No baseline models available"
        st.markdown(
            f'<div class="alert-info">'
            f'🏆 <strong>Best: {best["model"]}</strong> &nbsp;|&nbsp; '
            f'MAPE={best["MAPE"]:.2f}% &nbsp;|&nbsp; {msg}'
            f'</div>',
            unsafe_allow_html=True
)
        tab1,tab2 = st.tabs(["📋 Table","📊 Charts"])
        with tab1:
            st.dataframe(sub[["model","type","MAE","RMSE","MAPE"]]
                         .style.highlight_min(subset=["MAE","RMSE","MAPE"],color="#BBF7D0")
                         .format({"MAE":"{:.2f}","RMSE":"{:.2f}","MAPE":"{:.2f}"}),
                         use_container_width=True,height=400)
        with tab2:
            tc = {"Baseline":"#94A3B8","Exp. Smoothing":"#60A5FA","Multi-Seasonal":"#34D399",
                  "Statistical":COLORS["primary"],"Statistical/ML":COLORS["secondary"],"Deep Learning":COLORS["danger"]}
            bc = [tc.get(t,"#94A3B8") for t in sub["type"]]
            c1,c2,c3 = st.columns(3)
            for col,metric in zip([c1,c2,c3],["MAE","RMSE","MAPE"]):
                with col:
                    vals = sub[metric].tolist()
                    fig_b = go.Figure(go.Bar(x=list(range(len(sub))),y=vals,marker_color=bc,
                                             text=[f"{v:.1f}" for v in vals],textposition="outside"))
                    fig_b.update_layout(xaxis=dict(tickvals=list(range(len(sub))),
                                                    ticktext=sub["model"].tolist(),tickangle=30,tickfont=dict(size=8)),
                                         template="plotly_white",height=370,title=metric,showlegend=False,margin=dict(b=110))
                    bi = vals.index(min(vals))
                    fig_b.add_shape(type="rect",x0=bi-0.4,x1=bi+0.4,y0=0,y1=min(vals)*1.02,
                                     line=dict(color="gold",width=3))
                    st.plotly_chart(fig_b,use_container_width=True)
        

elif "Forecast" in page:
    st.markdown('<div class="page-title">🔮 UPI Forecast (Next Days)</div>', unsafe_allow_html=True)

    df = load_data()  # your existing loader
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # ── Controls ─────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        days = st.slider("Days to predict", 7, 30, 14)
    with col2:
        lag = st.selectbox("Seasonal Lag", [7, 14], index=0)

    # ── Forecast function ────────────────────
    def seasonal_naive(df, days, lag):
        preds = []
        last_date = df["Date"].max()

        for i in range(1, days + 1):
            target = last_date + pd.Timedelta(days=i)
            ref = target - pd.Timedelta(days=lag)

            ref_row = df[df["Date"] == ref]

            if not ref_row.empty:
                vol = ref_row["Volume (In Mn.)"].values[0]
                val = ref_row["Value (In Cr.)"].values[0]
            else:
                vol = df["Volume (In Mn.)"].iloc[-lag]
                val = df["Value (In Cr.)"].iloc[-lag]

            preds.append({
                "Date": target,
                "Pred_Volume": vol,
                "Pred_Value": val,
                "Day": target.day_name()
            })

        return pd.DataFrame(preds)

    pred_df = seasonal_naive(df, days, lag)

    # ── Alerts logic ─────────────────────────
    avg_vol = df["Volume (In Mn.)"].tail(30).mean()

    alerts = []
    for _, r in pred_df.iterrows():
        if r["Pred_Volume"] > avg_vol * 1.15:
            alerts.append(f"📈 High traffic expected on {r['Date'].date()} ({r['Day']})")
        elif r["Pred_Volume"] < avg_vol * 0.85:
            alerts.append(f"📉 Low activity expected on {r['Date'].date()} ({r['Day']})")

        if r["Day"] in ["Saturday", "Sunday"]:
            alerts.append(f"📅 Weekend pattern on {r['Date'].date()}")

    # ── Summary card ─────────────────────────
    st.markdown(
        f'<div class="alert-info">'
        f'📊 Avg Pred Volume: <strong>{pred_df["Pred_Volume"].mean():.2f}</strong> Mn &nbsp;|&nbsp; '
        f'📅 Horizon: <strong>{days} days</strong> &nbsp;|&nbsp; '
        f'🔁 Model: Seasonal Naive (lag={lag})'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Alerts display ───────────────────────
    if alerts:
        st.markdown('<div class="alert-trend"><strong>⚠️ Alerts</strong><br>' + "<br>".join(alerts[:6]) + '</div>', unsafe_allow_html=True)

    # ── Chart ────────────────────────────────
    fig = go.Figure()

    # past data
    fig.add_trace(go.Scatter(
        x=df["Date"].tail(60),
        y=df["Volume (In Mn.)"].tail(60),
        mode="lines",
        name="Actual"
    ))

    # forecast
    fig.add_trace(go.Scatter(
        x=pred_df["Date"],
        y=pred_df["Pred_Volume"],
        mode="lines+markers",
        name="Forecast"
    ))

    fig.update_layout(
        template="plotly_white",
        height=450,
        title="Volume Forecast",
        xaxis_title="Date",
        yaxis_title="Volume (Millions)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Table ────────────────────────────────
    st.dataframe(
        pred_df.assign(Date=pred_df["Date"].dt.strftime("%Y-%m-%d")),
        use_container_width=True
    )

    # ── Download ─────────────────────────────
    csv = pred_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Forecast", csv, "upi_forecast.csv", "text/csv")
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — AI REPORT (Ollama)
# ══════════════════════════════════════════════════════════════════════════════
elif "AI Report" in page:
    st.markdown('<div class="page-title">🤖 AI Report Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Generate professional monitoring reports using Gemini AI</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="alert-info">
    <strong>ℹ️ Cloud AI enabled:</strong><br>
    Reports are generated using Gemini API and work in Streamlit deployment.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2,1])

    with col1:
        report_type = st.selectbox(
            "Report Type",
            [
                "Daily Monitoring Summary",
                "Weekly Trend Analysis",
                "Festival Season Infrastructure Alert",
                "Monthly Performance Review",
                "YoY Growth Analysis",
                "Custom Prompt",
            ]
        )

        gemini_model = st.selectbox(
            "Gemini Model",
            [
                "gemini-2.5-flash",
                "gemini-2.5-pro"
            ]
        )

        n_days_report = st.slider(
            "Analysis window (days)",
            7,
            90,
            30
        )

    with col2:
        st.markdown("**Available AI Models:**")

        st.markdown(
            '<span style="background:#1A73E8;color:white;padding:2px 8px;border-radius:8px;font-size:11px;margin:2px;display:inline-block;">Gemini Flash</span>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<span style="background:#34A853;color:white;padding:2px 8px;border-radius:8px;font-size:11px;margin:2px;display:inline-block;">Gemini Pro</span>',
            unsafe_allow_html=True
        )

    # Custom prompt or auto-build
    if report_type == "Custom Prompt":

        custom_prompt = st.text_area(
            "Enter your prompt",
            height=150,
            placeholder="e.g. Analyse UPI transaction trends for November 2025 and identify infrastructure risks..."
        )

    else:

        risk_days = compute_risk_calendar(14)
        yoy_alerts = compute_yoy_alerts(df, target, n_days_report)

        type_prompts = {

            "Daily Monitoring Summary":
            build_report_prompt(
                df,
                target,
                7,
                risk_days,
                yoy_alerts
            ),

            "Weekly Trend Analysis":
            build_report_prompt(
                df,
                target,
                7,
                risk_days,
                yoy_alerts
            ),

            "Festival Season Infrastructure Alert":
            f"""You are an NPCI infrastructure analyst.

Write a 250-word alert report about upcoming festival season risks for UPI infrastructure.

Focus on these upcoming high-risk days:
{[r['date']+' ('+r['severity']+')' for r in risk_days[:5]]}

Include specific recommendations for:

- server scaling
- monitoring
- infrastructure readiness

Current daily average:
{df[target].tail(30).mean():.1f} {target}

Data through:
{df['Date'].max().strftime('%d %B %Y')}
""",

            "Monthly Performance Review":
            build_report_prompt(
                df,
                target,
                30,
                risk_days,
                yoy_alerts
            ),

            "YoY Growth Analysis":
            f"""You are an NPCI data analyst.

Write a 250-word year-over-year growth analysis for UPI {target}.

YoY alerts:
{[(a['date'], f"{a['raw_pct']:+.1f}%") for a in yoy_alerts[:5]]}

Latest value:
{df[target].iloc[-1]:.1f}

Annual totals:
{df.groupby('YearStr')[target].sum().round(0).to_dict()}

Data through:
{df['Date'].max().strftime('%d %B %Y')}
"""
        }

        custom_prompt = type_prompts.get(report_type, "")

        with st.expander("📋 View auto-generated prompt"):
            st.text(custom_prompt)

    if st.button("🤖 Generate Report",
                 type="primary",
                 use_container_width=True):

        with st.spinner("Generating report using Gemini..."):

            try:

                import google.generativeai as genai

                genai.configure(
                    api_key=st.secrets["GEMINI_API_KEY"]
                )

                model = genai.GenerativeModel(
                    gemini_model
                )

                final_prompt = f"""
You are an expert NPCI infrastructure analyst.

Generate a professional report.

{custom_prompt}
"""

                response = model.generate_content(
                    final_prompt
                )

                report = response.text

            except Exception as e:

                report = f"Error generating report: {str(e)}"

        st.markdown(
            '<div class="section-hdr">Generated Report</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="report-box">{report}</div>',
            unsafe_allow_html=True
        )

        st.download_button(
            "📥 Download Report",
            f"UPI {report_type}\nGenerated: {datetime.now().strftime('%d %b %Y %H:%M')}\n\n{report}",
            f"upi_report_{date.today().isoformat()}.txt",
            "text/plain"
        )
