"""
app.py
------
Real-Time Predictive Air Quality Dashboard (Capstone Project)

Architecture:
  Ingestion & Storage : ingestion.py (Open-Meteo Air Quality API -> DuckDB)
  Frontend & Analytics: this file (Streamlit + DuckDB SQL + Plotly + linear
                         forecast with shaded confidence intervals)

Run locally:
    streamlit run app.py

The app starts its own background ingestion thread on first load (see
`start_background_ingestion`), so a single `streamlit run` (or a single
Streamlit Community Cloud deployment) satisfies both the "Automated Data
Ingestion" and "Live Public Deployment" capstone requirements without
needing a second always-on process.
"""
#pip install duckdb
import threading
from datetime import timedelta
import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

import ingestion

# ---------------------------------------------------------------------------
# Page config + custom styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Air Quality Pulse | Real-Time Predictive Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main { background-color: #0E1117; }
    [data-testid="stMetricValue"] { font-size: 1.9rem; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem; opacity: 0.75; }
    .aqi-badge {
        display: inline-block; padding: 0.35rem 0.9rem; border-radius: 999px;
        font-weight: 700; font-size: 0.95rem; color: #0E1117;
    }
    .stApp header { background-color: rgba(0,0,0,0); }
    .block-container { padding-top: 1.6rem; }
    footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

POLLUTANT_LABELS = {
    "pm2_5": "PM2.5 (µg/m³)",
    "pm10": "PM10 (µg/m³)",
    "us_aqi": "US AQI",
    "european_aqi": "European AQI",
    "carbon_monoxide": "Carbon Monoxide (µg/m³)",
    "nitrogen_dioxide": "Nitrogen Dioxide (µg/m³)",
    "sulphur_dioxide": "Sulphur Dioxide (µg/m³)",
    "ozone": "Ozone (µg/m³)",
}

LOCATIONS = {
    "Nairobi, Kenya": (-1.2921, 36.8219),
    "London, UK": (51.5072, -0.1276),
    "New York, USA": (40.7128, -74.0060),
    "New Delhi, India": (28.6139, 77.2090),
    "Lagos, Nigeria": (6.5244, 3.3792),
    "Tokyo, Japan": (35.6762, 139.6503),
}

# ---------------------------------------------------------------------------
# 1. Automated background ingestion (runs once per app session/process)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def start_background_ingestion(latitude: float, longitude: float):
    """Launches ingestion.run_ingestion_loop() in a daemon thread. Cached by
    (lat, lon) so switching locations in the sidebar spins up a fresh poller
    without leaving the previous one orphaned mid-loop -- Streamlit tears
    down the old cached resource's closure once it's no longer referenced."""
    stop_event = threading.Event()
    thread = threading.Thread(
        target=ingestion.run_ingestion_loop,
        kwargs={
            "latitude": latitude,
            "longitude": longitude,
            "stop_event": stop_event,
        },
        daemon=True,
        name=f"ingestion-{latitude}-{longitude}",
    )
    thread.start()
    return {"thread": thread, "stop_event": stop_event}


# ---------------------------------------------------------------------------
# 2. In-memory analytical queries (DuckDB SQL)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=45, show_spinner=False)
def load_hourly_summary(db_path: str) -> pd.DataFrame:
    """Clean raw readings and summarize into hourly averages via DuckDB SQL."""
    try:
        con = duckdb.connect(db_path, read_only=True)
    except duckdb.IOException:
        return pd.DataFrame()
    try:
        query = f"""
            SELECT
                date_trunc('hour', utc_timestamp) AS hour,
                avg(pm2_5)            AS pm2_5,
                avg(pm10)             AS pm10,
                avg(us_aqi)           AS us_aqi,
                avg(european_aqi)     AS european_aqi,
                avg(carbon_monoxide)  AS carbon_monoxide,
                avg(nitrogen_dioxide) AS nitrogen_dioxide,
                avg(sulphur_dioxide)  AS sulphur_dioxide,
                avg(ozone)            AS ozone,
                count(*)              AS n_readings
            FROM {ingestion.TABLE_NAME}
            WHERE utc_timestamp >= now() - INTERVAL 7 DAY
            GROUP BY 1
            ORDER BY 1
        """
        df = con.execute(query).fetchdf()
        return df
    except duckdb.CatalogException:
        return pd.DataFrame()
    finally:
        con.close()


@st.cache_data(ttl=45, show_spinner=False)
def load_latest_reading(db_path: str) -> pd.DataFrame:
    try:
        con = duckdb.connect(db_path, read_only=True)
    except duckdb.IOException:
        return pd.DataFrame()
    try:
        query = f"""
            SELECT * FROM {ingestion.TABLE_NAME}
            ORDER BY utc_timestamp DESC LIMIT 1
        """
        return con.execute(query).fetchdf()
    except duckdb.CatalogException:
        return pd.DataFrame()
    finally:
        con.close()


# ---------------------------------------------------------------------------
# 3. Visual uncertainty forecast (linear trend + prediction interval)
# ---------------------------------------------------------------------------
def compute_forecast(
    df: pd.DataFrame, value_col: str, horizon_hours: int, confidence: float = 0.95
) -> pd.DataFrame:
    """Fits a linear trend to recent hourly averages and projects it forward,
    returning a point forecast plus a shaded confidence interval derived from
    the regression's prediction-interval formula."""
    data = df.dropna(subset=[value_col]).copy()
    if len(data) < 5:
        return pd.DataFrame()

    data = data.sort_values("hour")
    t0 = data["hour"].min()
    data["t"] = (data["hour"] - t0).dt.total_seconds() / 3600.0

    x = data["t"].to_numpy()
    y = data[value_col].to_numpy()
    n = len(x)
    dof = n - 2
    if dof < 1:
        return pd.DataFrame()

    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    residuals = y - y_hat
    se = np.sqrt(np.sum(residuals**2) / dof)
    x_mean = x.mean()
    sxx = np.sum((x - x_mean) ** 2)
    if sxx == 0:
        sxx = 1e-9
    t_val = stats.t.ppf((1 + confidence) / 2, dof)

    future_t = np.arange(x.max() + 1, x.max() + horizon_hours + 1)
    future_y = slope * future_t + intercept
    margin = t_val * se * np.sqrt(1 + 1 / n + (future_t - x_mean) ** 2 / sxx)

    # Physical, non-negative pollutant/AQI values -- clip the lower band at 0.
    lower = np.clip(future_y - margin, 0, None)
    upper = future_y + margin

    future_hours = [t0 + timedelta(hours=float(v)) for v in future_t]
    return pd.DataFrame(
        {"hour": future_hours, "forecast": future_y, "lower": lower, "upper": upper}
    )


def aqi_category(value: float) -> tuple[str, str]:
    """US AQI breakpoints -> (label, hex color)."""
    if value is None or np.isnan(value):
        return "Unknown", "#6c757d"
    if value <= 50:
        return "Good", "#4CAF50"
    if value <= 100:
        return "Moderate", "#FFEB3B"
    if value <= 150:
        return "Unhealthy (Sensitive)", "#FF9800"
    if value <= 200:
        return "Unhealthy", "#F44336"
    if value <= 300:
        return "Very Unhealthy", "#9C27B0"
    return "Hazardous", "#7B1E1E"


def build_forecast_chart(
    hist: pd.DataFrame, forecast: pd.DataFrame, value_col: str, label: str
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hist["hour"],
            y=hist[value_col],
            mode="lines+markers",
            name="Historical",
            line=dict(color="#3DB2FF", width=2),
            marker=dict(size=4),
        )
    )

    if not forecast.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast["hour"],
                y=forecast["upper"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast["hour"],
                y=forecast["lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(255, 179, 71, 0.22)",
                name="95% Confidence Interval",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast["hour"],
                y=forecast["forecast"],
                mode="lines+markers",
                name="Forecast",
                line=dict(color="#FFB347", width=2, dash="dash"),
                marker=dict(size=4, symbol="diamond"),
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis_title=label,
        xaxis_title="Time (UTC)",
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    location_name = st.selectbox("Location", list(LOCATIONS.keys()), index=0)
    latitude, longitude = LOCATIONS[location_name]

    pollutant = st.selectbox(
        "Metric to analyze",
        list(POLLUTANT_LABELS.keys()),
        format_func=lambda k: POLLUTANT_LABELS[k],
        index=0,
    )

    horizon = st.slider("Forecast horizon (hours)", min_value=3, max_value=48, value=12)
    confidence = st.select_slider(
        "Confidence level", options=[0.80, 0.90, 0.95, 0.99], value=0.95
    )

    auto_refresh = st.toggle("Auto-refresh page", value=False)
    refresh_seconds = st.number_input(
        "Refresh every (seconds)", min_value=15, max_value=600, value=60, step=15,
        disabled=not auto_refresh,
    )

    if st.button("🔄 Refresh data now", use_container_width=True):
        st.cache_data.clear()

    st.caption(
        "Data source: [Open-Meteo Air Quality API]"
        "(https://open-meteo.com/en/docs/air-quality-api) — free, no key required."
    )

if auto_refresh:
    st.markdown(
        f'<meta http-equiv="refresh" content="{int(refresh_seconds)}">',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Kick off ingestion for the selected location
# ---------------------------------------------------------------------------
job = start_background_ingestion(latitude, longitude)

DB_PATH = ingestion.DB_PATH

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# 🌍 Air Quality Pulse")
st.markdown(
    f"Real-time predictive dashboard for **{location_name}** — "
    "live ingestion, DuckDB analytics, and a shaded-uncertainty forecast."
)

hourly = load_hourly_summary(DB_PATH)
latest = load_latest_reading(DB_PATH)

if hourly.empty:
    st.info(
        "⏳ Collecting the first readings — the background ingestion thread just "
        "started and is backfilling recent history. This usually takes a few "
        "seconds; click **Refresh data now** shortly."
    )
    st.stop()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

current_val = latest[pollutant].iloc[0] if not latest.empty else np.nan
avg_24h = hourly.tail(24)[pollutant].mean()
min_7d = hourly[pollutant].min()
max_7d = hourly[pollutant].max()

with col1:
    st.metric(f"Current {POLLUTANT_LABELS[pollutant]}", f"{current_val:,.1f}" if pd.notna(current_val) else "—")
with col2:
    st.metric("24h Average", f"{avg_24h:,.1f}" if pd.notna(avg_24h) else "—")
with col3:
    st.metric("7-Day Min", f"{min_7d:,.1f}" if pd.notna(min_7d) else "—")
with col4:
    st.metric("7-Day Max", f"{max_7d:,.1f}" if pd.notna(max_7d) else "—")

if pollutant in ("us_aqi", "european_aqi") and pd.notna(current_val):
    label, color = aqi_category(current_val)
    st.markdown(
        f'<span class="aqi-badge" style="background-color:{color};">{label}</span>',
        unsafe_allow_html=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# Forecast chart
# ---------------------------------------------------------------------------
st.markdown("### 📈 Historical Trend & Forecast")
forecast_df = compute_forecast(hourly, pollutant, horizon, confidence)

if forecast_df.empty:
    st.warning(
        "Not enough historical readings yet to fit a reliable forecast "
        "(need at least 5 hourly points). Check back soon."
    )
    fig = build_forecast_chart(hourly, pd.DataFrame(), pollutant, POLLUTANT_LABELS[pollutant])
else:
    fig = build_forecast_chart(hourly, forecast_df, pollutant, POLLUTANT_LABELS[pollutant])

st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"Forecast method: ordinary least-squares linear trend fit on the last "
    f"{len(hourly)} hourly averages, projected {horizon}h forward with a "
    f"{int(confidence * 100)}% prediction interval "
    "(shaded band = wider uncertainty further from the last observation)."
)

# ---------------------------------------------------------------------------
# Raw / summarized data table
# ---------------------------------------------------------------------------
with st.expander("🔍 View underlying hourly summary (DuckDB query result)"):
    st.dataframe(
        hourly.sort_values("hour", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Architecture: Open-Meteo Air Quality API → Python ingestion client → "
    "DuckDB (`air_quality.duckdb`) → DuckDB SQL aggregation → Streamlit + Plotly."
)
