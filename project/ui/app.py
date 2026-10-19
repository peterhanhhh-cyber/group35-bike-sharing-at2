"""
Bike Sharing Demand Estimation — Streamlit prototype
41040 Introduction to Artificial Intelligence, AT2 AI Project, Group 35

Run from the project folder:
    streamlit run ui/app.py

The app loads the model selected in results/comparison_summary.json and reuses
the exact preprocessing defined in src/bike_common.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "ui", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import bike_common as bc
from model_defs import Autoencoder, CNNRegressor, LSTMRegressor, RegressionHead

st.set_page_config(
    page_title="Bike Sharing Demand Estimation",
    page_icon="🚲",
    layout="wide",
)

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday"]
WEATHER = {
    1: "1 — Clear / few clouds",
    2: "2 — Mist / cloudy",
    3: "3 — Light rain or snow",
    4: "4 — Heavy rain, snow or storm",
}


# ---------------------------------------------------------------------------
# Cached assets
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_grid() -> pd.DataFrame:
    return bc.build_hourly_grid()


@st.cache_resource(show_spinner=False)
def load_scaler() -> bc.FeatureScaler:
    return bc.FeatureScaler.load(ROOT / "models" / "scaler.joblib")


@st.cache_data(show_spinner=False)
def load_comparison() -> dict:
    path = ROOT / "results" / "comparison_summary.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


@st.cache_resource(show_spinner=False)
def load_model(name: str):
    """Return a dict describing the deployed artifact for one method."""
    if name == "Random Forest":
        model = joblib.load(ROOT / "models" / "random_forest.joblib")
        return {"kind": "rf", "model": model}

    if name == "LSTM":
        cfg = json.loads((ROOT / "results" / "lstm_config.json").read_text())
        model = LSTMRegressor(
            cfg["input_size"], cfg["hidden_size"], cfg["num_layers"],
            cfg["dropout"])
        state = torch.load(ROOT / "models" / "lstm_model.pt",
                           map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return {"kind": "lstm", "model": model, "config": cfg}

    if name == "2D CNN":
        cfg = json.loads((ROOT / "results" / "cnn_config.json").read_text())
        model = CNNRegressor(
            cfg["input_channels"], tuple(cfg["channels"]),
            cfg["kernel_size"], cfg["dropout"])
        state = torch.load(ROOT / "models" / "cnn_model.pt",
                           map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return {"kind": "cnn", "model": model, "config": cfg}

    if name == "AE + head":
        cfg = json.loads((ROOT / "results" / "ae_config.json").read_text())
        ae = Autoencoder(cfg["input_size"] if "input_size" in cfg
                         else len(cfg["features"]),
                         cfg["bottleneck"], cfg["hidden"])
        ae.encoder.load_state_dict(
            torch.load(ROOT / "models" / "ae_encoder.pt",
                       map_location="cpu", weights_only=True))
        head = RegressionHead(cfg["bottleneck"], cfg["head_hidden"])
        head.load_state_dict(
            torch.load(ROOT / "models" / "ae_head.pt",
                       map_location="cpu", weights_only=True))
        ae.eval()
        head.eval()
        return {"kind": "ae", "model": ae, "head": head, "config": cfg}

    raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
def season_for_month(month: int) -> int:
    if month in (12, 1, 2):
        return 4
    if month in (3, 4, 5):
        return 1
    if month in (6, 7, 8):
        return 2
    return 3


def find_anchor(grid: pd.DataFrame, month: int, hour: int, weekday: int,
                holiday: int, workingday: int):
    """Find the most recent observed hour matching the calendar scenario."""
    base = (grid["mnth"] == month) & (grid["hr"] == hour) & grid["observed"]
    attempts = [
        (base & (grid["weekday"] == weekday) & (grid["holiday"] == holiday)
         & (grid["workingday"] == workingday), "exact calendar match"),
        (base & (grid["weekday"] == weekday), "same weekday (holiday flag relaxed)"),
        (base, "same month and hour (weekday relaxed)"),
    ]
    for mask, quality in attempts:
        index = grid.index[mask]
        if len(index):
            return index.max(), quality, int(mask.sum())
    raise ValueError("No historical anchor found for this month and hour.")


def build_scenario_row(grid: pd.DataFrame, anchor: pd.Timestamp,
                       month: int, hour: int, weekday: int, holiday: int,
                       workingday: int, weathersit: int, temp: float,
                       atemp: float, hum: float, windspeed: float) -> pd.DataFrame:
    """Build the 23 shared features for the user's counterfactual scenario."""
    row = grid.loc[anchor].copy()
    row["season"] = season_for_month(month)
    row["mnth"] = month
    row["hr"] = hour
    row["weekday"] = weekday
    row["holiday"] = holiday
    row["workingday"] = workingday
    row["weathersit"] = weathersit
    row["temp"] = temp
    row["atemp"] = atemp
    row["hum"] = hum
    row["windspeed"] = windspeed

    base = bc.build_design_matrix(pd.DataFrame([row])).reset_index(drop=True)
    history = grid.loc[anchor, bc.HISTORY_COLUMNS].to_frame().T.reset_index(drop=True)
    return pd.concat([base, history], axis=1)[bc.FEATURE_COLUMNS].astype(float)


def build_window(grid: pd.DataFrame, anchor: pd.Timestamp,
                 scenario_row: pd.DataFrame, window: int = 168) -> pd.DataFrame:
    """The 168-hour feature window ending at the anchor, with the last hour
    replaced by the user's scenario."""
    position = grid.index.get_loc(anchor)
    positions = np.arange(position - (window - 1), position + 1)
    X = grid.iloc[positions][bc.FEATURE_COLUMNS].reset_index(drop=True).copy()
    X.iloc[-1] = scenario_row.iloc[0]
    return X


def predict_one(name: str, grid: pd.DataFrame, scaler: bc.FeatureScaler,
                anchor: pd.Timestamp, scenario_row: pd.DataFrame) -> float:
    assets = load_model(name)
    kind = assets["kind"]

    if kind == "rf":
        X = scaler.transform(scenario_row)
        return float(assets["model"].predict(X)[0])

    if kind == "ae":
        X = scaler.transform(scenario_row).astype(np.float32)
        with torch.no_grad():
            z = assets["model"].encoder(torch.tensor(X))
            scaled = assets["head"](z).item()
        cfg = assets["config"]
        return float(scaled * cfg["target_std"] + cfg["target_mean"])

    window = build_window(grid, anchor, scenario_row)
    X = scaler.transform(window)

    if kind == "lstm":
        tensor = torch.tensor(X[None, :, :], dtype=torch.float32)
        with torch.no_grad():
            scaled = assets["model"](tensor).item()
        cfg = assets["config"]
        return float(scaled * cfg["target_std"] + cfg["target_mean"])

    if kind == "cnn":
        image = bc.reshape_weekly(X[None, :, :]).astype(np.float32)
        with torch.no_grad():
            scaled = assets["model"](torch.tensor(image)).item()
        cfg = assets["config"]
        return float(scaled * cfg["target_std"] + cfg["target_mean"])

    raise ValueError(f"Unknown model kind: {kind}")


# ---------------------------------------------------------------------------
# Sidebar and header
# ---------------------------------------------------------------------------
comparison = load_comparison()
selected = comparison.get("selected_model", "Random Forest")
selected_table = comparison.get("table", {}).get(selected, {})

st.title("Bike Sharing Demand Estimation")
st.caption("Historical Data Prototype | Input a scenario to view estimated "
           "total rentals for that time slot")
st.write(
    "This prototype estimates the **total number of rentals** in one hour "
    "from the calendar and weather scenario. It is trained on the UCI "
    "Capital Bikeshare data for 2011–2012 and is intended to show how the "
    "selected model behaves, not to forecast current demand in Sydney."
)

with st.sidebar:
    st.header("Group 35")
    st.write("41040 Introduction to AI — AT2 AI Project")
    st.success(f"Selected model: **{selected}**")
    if selected_table:
        st.metric("Validation MAE", f"{selected_table['val MAE']:.1f} rentals/h")
        st.metric("Test MAE", f"{selected_table['test MAE']:.1f} rentals/h")
        st.metric("Test R²", f"{selected_table['test R2']:.3f}")
    st.divider()
    compare_all = st.checkbox("Compare all four methods", value=False,
                              help="Show the estimate from every candidate model.")
    st.divider()
    st.caption(
        "Prototype limits: the model estimates demand, not bike or dock "
        "availability. The history features are auto-filled from the "
        "selected historical anchor, so changing the weather does not "
        "change the recorded past demand."
    )


# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------
DEMOS = {
    "Custom scenario": {},
    "Weekday morning commute": dict(
        month=6, hour=8, weekday=1, holiday=False, workingday=True,
        weathersit=1, temp=0.55, atemp=0.52, hum=0.70, windspeed=0.15),
    "Weekend afternoon": dict(
        month=7, hour=15, weekday=6, holiday=False, workingday=False,
        weathersit=1, temp=0.70, atemp=0.68, hum=0.50, windspeed=0.20),
    "Boundary values (zero weather)": dict(
        month=1, hour=0, weekday=3, holiday=False, workingday=True,
        weathersit=4, temp=0.0, atemp=0.0, hum=0.0, windspeed=0.0),
}

DEFAULTS = dict(month=6, hour=8, weekday=1, holiday=False, workingday=True,
                weathersit=1, temp=0.55, atemp=0.52, hum=0.70, windspeed=0.15)
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

demo = st.selectbox("Load an example scenario", list(DEMOS))
if st.session_state.get("_demo_applied") != demo:
    for key, value in DEMOS[demo].items():
        st.session_state[key] = value
    st.session_state["_demo_applied"] = demo

grid = load_grid()
scaler = load_scaler()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
with st.form("scenario_form"):
    left, middle, right = st.columns(3)

    with left:
        st.subheader("1. Time")
        month = st.selectbox("Month", range(1, 13), key="month",
                             format_func=lambda m: MONTHS[m - 1])
        hour = st.slider("Hour of day", 0, 23, key="hour")
        weekday = st.selectbox("Day of week", range(7), key="weekday",
                               format_func=lambda d: WEEKDAYS[d])

    with middle:
        st.subheader("2. Calendar")
        holiday = st.checkbox("Public holiday", key="holiday")
        workingday = st.checkbox("Working day", key="workingday")
        if holiday and workingday:
            st.warning("A day is not normally both a public holiday and a "
                       "working day. The estimate will still run.")
        st.subheader("3. Weather")
        weathersit = st.selectbox("Weather category", [1, 2, 3, 4], key="weathersit",
                                  format_func=lambda w: WEATHER[w])

    with right:
        st.subheader("Weather values")
        temp = st.slider("Temperature (normalised 0–1)", 0.0, 1.0, key="temp",
                         step=0.01,
                         help=f"≈ {47 * st.session_state.temp - 8:.1f} °C "
                              "using the dataset's normalisation")
        atemp = st.slider("Feels-like temperature (0–1)", 0.0, 1.0, key="atemp",
                          step=0.01,
                          help=f"≈ {66 * st.session_state.atemp - 16:.1f} °C")
        hum = st.slider("Humidity (0–1)", 0.0, 1.0, key="hum", step=0.01,
                        help=f"≈ {100 * st.session_state.hum:.0f}%")
        windspeed = st.slider("Wind speed (0–1)", 0.0, 1.0, key="windspeed",
                              step=0.01,
                              help=f"≈ {67 * st.session_state.windspeed:.0f} km/h")

    with st.expander("Validation demo — enter manual weather values"):
        st.caption("Use this to demonstrate missing or out-of-range input. "
                   "Leave a field blank to trigger the missing-input check.")
        manual_mode = st.checkbox("Use manual values instead of the sliders",
                                  value=False)
        m1, m2, m3 = st.columns(3)
        manual_temp = m1.text_input("Temperature (0–1)", value="0.55")
        manual_hum = m2.text_input("Humidity (0–1)", value="0.65")
        manual_wind = m3.text_input("Wind speed (0–1)", value="0.15")

    submitted = st.form_submit_button("Estimate total rentals", type="primary")

# ---------------------------------------------------------------------------
# Validation and prediction
# ---------------------------------------------------------------------------
def parse_manual():
    if not manual_mode:
        return st.session_state.temp, st.session_state.hum, st.session_state.windspeed, None
    values = {}
    for label, raw in [("temperature", manual_temp), ("humidity", manual_hum),
                       ("wind speed", manual_wind)]:
        if raw is None or not raw.strip():
            return None, None, None, f"Missing input: please enter a {label} value."
        try:
            value = float(raw)
        except ValueError:
            return None, None, None, f"Invalid input: the {label} must be a number."
        if not 0.0 <= value <= 1.0:
            return None, None, None, (f"Out of range: the {label} must be between "
                                      f"0 and 1 (got {value}).")
        values[label.split()[0]] = value
    return values["temperature"], values["humidity"], values["wind"], None

if submitted:
    temp_used, hum_used, wind_used, error = parse_manual()
    if error:
        st.error(error)
        st.session_state.pop("estimate", None)
    else:
        anchor, quality, n_matches = find_anchor(
            grid, month, hour, weekday, int(holiday), int(workingday))
        scenario_row = build_scenario_row(
            grid, anchor, month, hour, weekday, int(holiday), int(workingday),
            weathersit, temp_used, st.session_state.atemp, hum_used, wind_used)
        estimate = max(0.0, predict_one(selected, grid, scaler, anchor, scenario_row))
        st.session_state["estimate"] = {
            "value": estimate,
            "anchor": anchor,
            "quality": quality,
            "matches": n_matches,
            "scenario_row": scenario_row,
            "inputs": dict(month=month, hour=hour, weekday=weekday,
                           holiday=holiday, workingday=workingday,
                           weathersit=weathersit, temp=temp_used,
                           atemp=st.session_state.atemp, hum=hum_used,
                           windspeed=wind_used),
        }

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
if "estimate" in st.session_state:
    result = st.session_state["estimate"]
    anchor = result["anchor"]
    estimate = result["value"]
    actual = grid.loc[anchor, "cnt"]

    st.divider()
    out_left, out_right = st.columns([1, 1.6])
    with out_left:
        st.subheader("Estimated demand")
        st.metric("Estimated total rentals", f"{estimate:,.0f}")
        st.caption(f"Model: {selected} | History anchor: "
                   f"{anchor.strftime('%Y-%m-%d %H:%M')}")
        if pd.notna(actual):
            st.caption(f"Observed rentals at the anchor: {actual:,.0f}")
        st.info(
            "Prototype based on 2011–2012 historical data. The estimate is a "
            "demand scenario, not a live forecast or a bike-availability check."
        )
        edge = [
            label for label, value in (
                ("temperature", result["inputs"]["temp"]),
                ("feels-like temperature", result["inputs"]["atemp"]),
                ("humidity", result["inputs"]["hum"]),
                ("wind speed", result["inputs"]["windspeed"]),
            )
            if value <= 0.001 or value >= 0.999
        ]
        if edge:
            st.warning(
                "Edge-of-data warning: " + ", ".join(edge) +
                " sits at the boundary of the observed range. Treat the "
                "estimate as an extrapolation."
            )
        if result["quality"] != "exact calendar match":
            st.warning(f"History anchor match: {result['quality']} "
                       f"({result['matches']} candidate hours).")

    with out_right:
        st.subheader("Recent demand used as context")
        history = grid.loc[anchor - pd.Timedelta(hours=167):anchor, "cnt"]
        fig, ax = plt.subplots(figsize=(7.4, 3.1))
        ax.plot(history.index, history.values, color="#1f4e79", lw=1.2,
                label="observed rentals (last 7 days)")
        ax.scatter([anchor], [estimate], color="#c00000", s=45, zorder=5,
                   label="scenario estimate")
        ax.set(ylabel="Rentals per hour")
        ax.legend(loc="upper left", fontsize=8)
        ax.tick_params(axis="x", labelrotation=20)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("Features used for this estimate"):
        st.dataframe(result["scenario_row"].T.rename(columns={0: "value"}).round(4))

    if compare_all:
        st.subheader("All four candidate methods")
        rows = {}
        for name in ("Random Forest", "LSTM", "2D CNN", "AE + head"):
            try:
                value = max(0.0, predict_one(name, grid, scaler, anchor,
                                             result["scenario_row"]))
                rows[name] = value
            except Exception as exc:  # a missing artifact should not break the page
                rows[name] = np.nan
                st.caption(f"{name}: not available ({exc})")
        compare_frame = pd.Series(rows, name="estimate").to_frame()
        st.dataframe(compare_frame.style.format("{:.0f}"))
        fig, ax = plt.subplots(figsize=(6.4, 2.8))
        ax.bar(compare_frame.index, compare_frame["estimate"], color="#5b9bd5")
        ax.set(title="Scenario estimate by method", ylabel="Rentals per hour")
        ax.tick_params(axis="x", labelrotation=15)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
else:
    st.info("Choose a scenario and press **Estimate total rentals**.")

# ---------------------------------------------------------------------------
# Tabs: comparison and explanation
# ---------------------------------------------------------------------------
tab_compare, tab_about = st.tabs(["Model comparison", "How it works"])

with tab_compare:
    if comparison.get("table"):
        st.subheader("Four-way comparison on the frozen test period")
        table = pd.DataFrame(comparison["table"]).T
        numeric_cols = ["val MAE", "val RMSE", "val R2", "test MAE",
                        "test RMSE", "test R2", "train seconds", "model MB",
                        "single-row ms"]
        available = [c for c in numeric_cols if c in table.columns]
        st.dataframe(table[available].astype(float).round(3))
        fig, ax = plt.subplots(figsize=(7.4, 3.2))
        order = table["val MAE"].astype(float).sort_values().index
        x = np.arange(len(order))
        width = 0.36
        ax.bar(x - width / 2, table.loc[order, "val MAE"].astype(float),
               width, label="validation", color="#8faadc")
        ax.bar(x + width / 2, table.loc[order, "test MAE"].astype(float),
               width, label="test", color="#1f4e79")
        ax.set_xticks(x, order)
        ax.set(ylabel="MAE (rentals/hour)", title="MAE by method")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(comparison.get("selection_reason", ""))
    else:
        st.info("Run the comparison notebook to populate this tab.")

with tab_about:
    st.subheader("What the system does")
    st.markdown(
        """
        1. **Input** — month, hour, weekday, holiday/working-day flags and a
           weather scenario.
        2. **Anchor** — the app finds the most recent observed hour in the
           dataset with the same calendar profile and retrieves its past
           demand (previous hour, yesterday, last week, 24-hour and 168-hour
           means).
        3. **Feature construction** — the 18 calendar/weather features are
           built from the scenario, and the 5 history features come from the
           anchor. This is the same 23-feature matrix used in every notebook.
        4. **Prediction** — the selected model produces an hourly rental
           estimate, which is displayed with the historical context.

        **Why the history features?** They tell the model the local demand
        level, which reduces the 2011–2012 distribution shift. They are all
        strictly past values, so the estimate remains causal.

        **Limitations**
        * The prototype estimates demand, not bike or dock availability.
        * The anchor is historical; changing the weather does not rewrite the
          recorded past demand.
        * The model is trained on Washington, D.C. data and is not valid for
          Sydney without retraining.
        * There is no live weather API and no uncertainty interval in this
          first version.
        """
    )
