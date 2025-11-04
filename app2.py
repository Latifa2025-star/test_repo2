import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="NASA GISTEMP — Interactive", layout="wide")

@st.cache_data(show_spinner=False)
def load_data():
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
    df = pd.read_csv(url, skiprows=1)
    df = df.rename(columns=lambda c: str(c).strip())
    df = df[df["Year"].astype(str).str.match(r"^\d{4}$", na=False)]
    df["Year"] = df["Year"].astype(int)

    # Create Anomaly from NASA's annual column ('J-D')
    jd_col = next((c for c in df.columns if str(c).strip().startswith("J-D")), None)
    if jd_col is None:
        raise RuntimeError("Could not find the NASA annual anomaly column 'J-D'.")
    df["Anomaly"] = pd.to_numeric(df[jd_col], errors="coerce")
    df = df.dropna(subset=["Anomaly"]).copy()
    df = df[df["Year"] >= 1880].reset_index(drop=True)

    # Optional seasonal columns for future plots
    for c in ["DJF", "MAM", "JJA", "SON"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df = load_data()

st.title("NASA GISTEMP: Global Temperature Anomaly (1880–present)")
st.caption("Source: NASA GISS GISTEMP v4. Baseline ≈ 1951–1980.")

# ---- Sidebar controls ----
years = df["Year"].tolist()
y_min, y_max = min(years), max(years)

start, end = st.sidebar.slider(
    "Year range",
    min_value=int(y_min),
    max_value=int(y_max),
    value=(int(y_min), int(y_max)),
    step=1,
)

smoothing = st.sidebar.selectbox(
    "Smoothing window",
    options=[0, 5, 10],
    index=2,
    format_func=lambda k: "None" if k == 0 else f"{k}-year",
)

threshold = st.sidebar.selectbox(
    "Show threshold line",
    options=[None, 1.5, 2.0],
    index=1,
    format_func=lambda v: "None" if v is None else f"+{v} °C",
)

# ---- Filter + smooth ----
d = df[(df["Year"] >= start) & (df["Year"] <= end)].copy()
if smoothing and smoothing > 1:
    d["Smoothed"] = d["Anomaly"].rolling(window=int(smoothing), center=True, min_periods=1).mean()
else:
    d["Smoothed"] = np.nan

# ---- Figure ----
fig = go.Figure()
fig.add_trace(go.Scatter(x=d["Year"], y=d["Anomaly"], mode="lines",
                         name="Annual", line=dict(width=2)))
if d["Smoothed"].notna().any():
    fig.add_trace(go.Scatter(x=d["Year"], y=d["Smoothed"], mode="lines",
                             name=f"{smoothing}-year smooth", line=dict(width=3)))

# Baseline + optional threshold
fig.add_hline(y=0, line_dash="dash")
if threshold is not None:
    fig.add_hline(y=float(threshold), line_dash="dot")

# annotate last point
last = d.iloc[-1]
fig.add_trace(go.Scatter(
    x=[last["Year"]], y=[last["Anomaly"]],
    mode="markers+text", text=[f"{last['Anomaly']:.2f}°C"],
    textposition="top center", name="Latest year"
))

fig.update_layout(
    template="plotly_white",
    width=None, height=500,
    legend=dict(orientation="h", x=0, y=1.1),
    xaxis_title="Year",
    yaxis_title="Temperature Anomaly (°C)",
    title=f"Global Temperature Anomaly ({start}–{end})"
)

col1, col2 = st.columns([3,1], gap="large")
with col1:
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.metric("Last year", int(df["Year"].max()))
    st.metric("Last anomaly (°C)", f"{float(df.loc[df['Year'].idxmax(),'Anomaly']):.2f}")
    st.write("**Notes**")
    st.write("- Anomaly relative to ~1951–1980 mean.")
    st.write("- Data: NASA GISS, GISTEMP v4.")

