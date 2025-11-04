# === Cell 1: Setup & Imports ===
# (Run this once per session)
!pip -q install plotly>=5.22 ipywidgets>=8.1
!pip install prophet

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from ipywidgets import widgets, HBox, VBox, Layout, interactive_output
from IPython.display import display, Markdown

# Render Plotly in notebook/Colab
pio.renderers.default = "colab"

# Enable the widget manager in Colab (safe if not in Colab)
try:
    from google.colab import output
    output.enable_custom_widget_manager()
except Exception:
    pass

print("Setup complete.")

# === Cell 2: Load NASA GISTEMP global temperature anomaly data ===
# Source: https://data.giss.nasa.gov/gistemp/
# Direct CSV: https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv

import io, requests
from google.colab import drive

drive.mount('/content/drive')
df = pd.read_csv("/content/drive/MyDrive/Python_Pizza_Event/GLB.Ts+dSST.csv", skiprows=1)
df = df.rename(columns=lambda c: c.strip())
df = df[df["Year"].astype(str).str.match(r"^\d{4}$", na=False)]
data = df
df["Year"] = df["Year"].astype(int)
df["J-D"] = pd.to_numeric(df["J-D"], errors="coerce")
df = df.dropna(subset=["J-D"]).copy()
df = df[df["Year"] >= 1880]
df = df.rename(columns={"J-D": "Anomaly"})
df = df[["Year","Anomaly"]].reset_index(drop=True)
print(df.head(10))

# === Cell 3: Quick EDA & Context ===
print("Data rows:", len(df), "| Year range:", df['Year'].min(), "→", df['Year'].max())
display(df.describe().T)

# Rolling means for optional smoothing
df['Anomaly_5yr'] = df['Anomaly'].rolling(window=5, center=True, min_periods=1).mean()
df['Anomaly_10yr'] = df['Anomaly'].rolling(window=10, center=True, min_periods=1).mean()

last_year = int(df['Year'].max())
last_val = float(df.loc[df['Year']==last_year, 'Anomaly'].iloc[0])
Markdown(f"""**Latest (~{last_year}) anomaly:** {last_val:.2f} °C relative to 1951–1980 baseline.""")
# === Cell 4: Absolute temperature series from anomalies ===
BASELINE_C = 14.0  # 1951–1980 global mean (approx) by NASA
df["Absolute_Temp"] = BASELINE_C + df["Anomaly"]

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Year"], y=df["Absolute_Temp"], mode="lines",
                         name="Global mean temperature (approx)"))
fig.add_hline(y=BASELINE_C, line_dash="dash", annotation_text="1951–1980 mean ≈ 14.0°C")
fig.update_layout(title="Observed Global Temperature (approx.)",
                  xaxis_title="Year", yaxis_title="Temperature (°C)",
                  template="plotly_white", width=900, height=450)
fig.show()

# === Cell 7: Mini Interactive Dashboard (Widgets) ===
years = sorted(df['Year'].unique().tolist())

w_year_min = widgets.IntSlider(min=y_min, max=y_max, value=y_min, description="Start")
w_year_max = widgets.IntSlider(min=y_min, max=y_max, value=y_max, description="End")
w_year_min.continuous_update = False
w_year_max.continuous_update = False
w_smooth = widgets.Dropdown(options=[("None",0),("5-year",5),("10-year",10)], value=10, description="Smoothing")
w_threshold = widgets.Dropdown(options=[("None", None), ("+1.5°C", 1.5), ("+2.0°C", 2.0)], value=1.5, description="Threshold")

out = widgets.Output()

years = sorted(df['Year'].unique().tolist())
y_min, y_max = years[0], years[-1]

def _to_valid_year(x):
    # Handle floats/strings cleanly and clamp to [y_min, y_max]
    try:
        x = int(round(float(x)))
    except Exception:
        x = y_min
    return max(y_min, min(x, y_max))

def render_dashboard(start, end, smooth_k, thresh):
    with out:
        out.clear_output(wait=True)

        # Coerce & sanitize years
        start = _to_valid_year(start)
        end   = _to_valid_year(end)
        if start > end:
            start, end = end, start  # auto-fix reversed range

        d = df[(df['Year'] >= start) & (df['Year'] <= end)].copy()

        # Safe if empty
        if d.empty:
            print(f"No data between {start} and {end}.")
            return

        # Smoothing
        if smooth_k and smooth_k > 1:
            d['Smoothed'] = d['Anomaly'].rolling(window=int(smooth_k), center=True, min_periods=1).mean()
        else:
            d['Smoothed'] = np.nan

        # Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d['Year'], y=d['Anomaly'], mode='lines',
                                 name='Annual', line=dict(color = "firebrick", width=2)))

        if d['Smoothed'].notna().any():
            fig.add_trace(go.Scatter(x=d['Year'], y=d['Smoothed'], mode='lines',
                                     name=f'{smooth_k}-year smooth', line=dict(color = "black", width=3)))

        fig.add_hline(y=0, line_dash='dash')
        if thresh is not None:
            fig.add_hline(y=float(thresh), line_dash='dot')

        # Only annotate if we have at least one row
        last_row = d.iloc[-1]
        fig.add_trace(go.Scatter(
            x=[last_row['Year']], y=[last_row['Anomaly']],
            mode='markers+text', marker=dict(size=8),
            text=[f"{last_row['Anomaly']:.2f}°C"], textposition="top center",
            name="Latest year"
        ))

        fig.update_layout(
            title="Interactive Fever Chart — Explore Years, Smoothing, and Thresholds",
            xaxis_title="Year", yaxis_title="Temperature Anomaly (°C)",
            template="plotly_white", width=900, height=480,
            legend=dict(orientation="h", x=0, y=1.1)
        )
        fig.show()

controls = {"start": w_year_min, "end": w_year_max, "smooth_k": w_smooth, "thresh": w_threshold}
io = interactive_output(render_dashboard, controls)

display(VBox([
    HBox([w_year_min, w_year_max]),
    HBox([w_smooth, w_threshold]),
    out
]), io)
