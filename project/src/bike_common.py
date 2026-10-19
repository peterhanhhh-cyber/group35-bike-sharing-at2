"""
Shared data, feature and evaluation utilities for the 41040 AT2 bike-sharing
demand prediction project (Group 35).

The four member notebooks import this module so that every method uses exactly
the same rows, target, feature definitions and metrics.  Keeping these pieces in
one place is what makes the four-way comparison fair.

Run from anywhere inside the project tree: the data directory is located by
walking upwards until ``hour.csv`` is found.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def find_data_dir(start: Path | str | None = None) -> Path:
    """Walk upwards from *start* (then from this file) until hour.csv is found."""
    here = Path(start or Path.cwd()).resolve()
    module_dir = Path(__file__).resolve().parent
    for candidate in (here, *here.parents, module_dir, *module_dir.parents):
        if (candidate / "hour.csv").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find hour.csv in this folder or any parent folder. "
        "Run the notebook from inside the AI project directory."
    )


DATA_DIR = find_data_dir()
PROJECT_DIR = DATA_DIR / "project"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
MODELS_DIR = PROJECT_DIR / "models"
for _d in (RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
RAW_COLUMNS = [
    "instant", "dteday", "season", "yr", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "temp", "atemp", "hum", "windspeed",
    "casual", "registered", "cnt",
]

TARGET = "cnt"

# Base feature columns produced by build_design_matrix (18 columns).
BASE_FEATURE_COLUMNS = [
    "season_1", "season_2", "season_3",
    "weather_1", "weather_2", "weather_3",
    "mnth_sin", "mnth_cos",
    "hr_sin", "hr_cos",
    "wd_sin", "wd_cos",
    "holiday", "workingday",
    "temp", "atemp", "hum", "windspeed",
]

# Causal demand-history columns added on the complete hourly grid.  Every value
# is computed from hours strictly before the prediction hour, so it is
# available to the user interface at prediction time.
HISTORY_COLUMNS = ["lag1", "lag24", "lag168", "roll24", "roll168"]

# The full shared feature set used by all four methods (23 columns).
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + HISTORY_COLUMNS

# Warm-up rows: a full week of history is needed before lag168/roll168 exist.
WARMUP_HOURS = 168
# First grid position used as a modelling sample.  It is the earliest end
# point at which a complete 168-hour window contains only defined features.
MIN_POSITION = 2 * WARMUP_HOURS - 1  # 335

# Continuous columns that are standardised.  One-hot indicator columns are
# already on a 0/1 scale but are standardised too so the autoencoder sees a
# common scale; this does not change tree or linear predictions.
SCALE_COLUMNS = FEATURE_COLUMNS[:]


def load_raw() -> pd.DataFrame:
    """Load hour.csv with a parsed date and a proper datetime index column."""
    df = pd.read_csv(DATA_DIR / "hour.csv", parse_dates=["dteday"])
    df["dt"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    return df


def load_split_files() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the frozen chronological train / validation / test splits."""
    parts = {}
    for name in ("train", "validation", "test"):
        part = pd.read_csv(DATA_DIR / f"{name}.csv", parse_dates=["dteday"])
        part["dt"] = part["dteday"] + pd.to_timedelta(part["hr"], unit="h")
        parts[name] = part
    return parts["train"], parts["validation"], parts["test"]


def check_split_integrity() -> dict:
    """Cheap sanity checks that the frozen splits are what the plan promised."""
    manifest = pd.read_csv(DATA_DIR / "split_manifest.csv", parse_dates=["dteday"])
    train, val, test = load_split_files()
    rows = {
        "train": len(train),
        "validation": len(val),
        "test": len(test),
    }
    overlaps = {
        "train_val": len(set(train.instant) & set(val.instant)),
        "train_test": len(set(train.instant) & set(test.instant)),
        "val_test": len(set(val.instant) & set(test.instant)),
    }
    return {
        "rows": rows,
        "manifest_rows": int(len(manifest)),
        "all_rows_accounted_for": int(sum(rows.values())) == int(len(manifest)),
        "overlaps": overlaps,
        "train_range": [str(train.dt.min()), str(train.dt.max())],
        "validation_range": [str(val.dt.min()), str(val.dt.max())],
        "test_range": [str(test.dt.min()), str(test.dt.max())],
        "split_in_manifest": manifest.groupby("split").size().to_dict(),
        "duplicate_instants": int(manifest.instant.duplicated().sum()),
    }


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def build_design_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Turn raw calendar + weather columns into the 18 shared model features.

    Encodings:
      * season and weathersit are one-hot encoded, with the last level
        (season 4 / weathersit 4) as the reference category.
      * month, hour and weekday are encoded cyclically (sin/cos) so that
        23:00 is close to 00:00 and December is close to January.
      * holiday and workingday stay as 0/1 flags.
      * temp, atemp, hum and windspeed remain numeric and are standardised
        later, using training statistics only.

    No demand-derived features (casual, registered, lagged cnt) are used.
    This keeps the input available at prediction time and avoids target
    leakage.
    """
    out = pd.DataFrame(index=frame.index)

    season = frame["season"].astype(int)
    weather = frame["weathersit"].astype(int)
    for level in (1, 2, 3):
        out[f"season_{level}"] = (season == level).astype(float)
        out[f"weather_{level}"] = (weather == level).astype(float)

    month = frame["mnth"].astype(float)
    hour = frame["hr"].astype(float)
    weekday = frame["weekday"].astype(float)

    out["mnth_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    out["mnth_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    out["hr_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hr_cos"] = np.cos(2 * np.pi * hour / 24)
    out["wd_sin"] = np.sin(2 * np.pi * weekday / 7)
    out["wd_cos"] = np.cos(2 * np.pi * weekday / 7)

    out["holiday"] = frame["holiday"].astype(float)
    out["workingday"] = frame["workingday"].astype(float)

    for col in ("temp", "atemp", "hum", "windspeed"):
        out[col] = frame[col].astype(float)

    return out[BASE_FEATURE_COLUMNS]


class FeatureScaler:
    """Standardise features using training rows only (no leakage)."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X: pd.DataFrame) -> "FeatureScaler":
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() on the training features first.")
        return self.scaler.transform(X)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: Path) -> None:
        import joblib

        joblib.dump(self.scaler, path)

    @classmethod
    def load(cls, path: Path) -> "FeatureScaler":
        import joblib

        obj = cls()
        obj.scaler = joblib.load(path)
        obj.fitted = True
        return obj


def prepare_tabular() -> dict:
    """
    Load the frozen splits and return scaled tabular arrays plus metadata.

    Every model notebook calls this function, so the train/validation/test
    rows, the shared feature set and the scaling are identical across the four
    methods.

    The first two weeks of the dataset are discarded for *all* methods.  The
    first week is needed before ``lag168``/``roll168`` exist; the second week
    is needed so that a full 168-hour sliding window contains no warm-up NaNs.
    Validation and test start long after this point, so they are unaffected.
    """
    train, val, test = load_split_files()
    grid = build_hourly_grid()
    positions = {name: grid.index.get_indexer(part["dt"])
                 for name, part in
                 (("train", train), ("validation", val), ("test", test))}

    def collect(part: pd.DataFrame, pos: np.ndarray):
        frame = grid.iloc[pos][FEATURE_COLUMNS].reset_index(drop=True)
        mask = (pos >= MIN_POSITION) & frame.notna().all(axis=1).to_numpy()
        return (part.loc[mask].reset_index(drop=True),
                frame.loc[mask].reset_index(drop=True),
                pos[mask])

    train, Xtr_raw, ptr = collect(train, positions["train"])
    val, Xva_raw, pva = collect(val, positions["validation"])
    test, Xte_raw, pte = collect(test, positions["test"])

    scaler = FeatureScaler().fit(Xtr_raw)
    return {
        "train": train,
        "validation": val,
        "test": test,
        "positions": {"train": ptr, "validation": pva, "test": pte},
        "X_train": scaler.transform(Xtr_raw),
        "X_validation": scaler.transform(Xva_raw),
        "X_test": scaler.transform(Xte_raw),
        "X_train_raw": Xtr_raw,
        "X_validation_raw": Xva_raw,
        "X_test_raw": Xte_raw,
        "y_train": train[TARGET].to_numpy(dtype=float),
        "y_validation": val[TARGET].to_numpy(dtype=float),
        "y_test": test[TARGET].to_numpy(dtype=float),
        "scaler": scaler,
        "feature_names": FEATURE_COLUMNS,
    }


def build_hourly_grid(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Reindex the data onto a complete hourly grid and fill the 165 missing
    hours using *past-only* forward fill.

    The UCI export has 165 missing hours (0.95%).  Calendar fields are rebuilt
    exactly from the timestamp; date-level fields come from a daily table;
    weather fields are forward-filled.  No future value is used, so sliding
    windows built on this grid remain causal.

    Returns a frame indexed by datetime with raw columns, the shared design
    matrix columns, and the target (NaN where the hour was not observed).
    """
    if raw is None:
        raw = load_raw()

    full_hours = pd.date_range(raw["dt"].min(), raw["dt"].max(), freq="h")
    max_date = (full_hours.max() + pd.Timedelta(days=1)).normalize()
    full_days = pd.date_range(raw["dt"].min().normalize(), max_date - pd.Timedelta(days=1), freq="D")

    daily = (
        raw.groupby("dteday")
        .agg(season=("season", "first"), yr=("yr", "first"),
             holiday=("holiday", "max"), workingday=("workingday", "max"))
        .reindex(full_days)
        .ffill()
    )
    # The first date is always present, so ffill fully covers the range.

    weather = (
        raw.set_index("dt")[["weathersit", "temp", "atemp", "hum", "windspeed"]]
        .reindex(full_hours)
        .ffill()
    )
    # The target is *not* forward-filled: a missing hour stays missing so that
    # it can never become a training or evaluation sample.
    target = raw.set_index("dt")["cnt"].reindex(full_hours)

    grid = pd.DataFrame(index=full_hours)
    grid["dteday"] = grid.index.normalize()
    grid["mnth"] = grid.index.month
    grid["hr"] = grid.index.hour
    grid["weekday"] = (grid.index.dayofweek + 1) % 7  # UCI: Sunday=0
    grid = grid.join(daily, on="dteday")
    grid = grid.join(weather)
    grid["cnt"] = target
    grid["season"] = grid["season"].astype(int)
    grid["holiday"] = grid["holiday"].astype(int)
    grid["workingday"] = grid["workingday"].astype(int)
    grid["weathersit"] = grid["weathersit"].astype(int)

    design = build_design_matrix(grid)
    for col in BASE_FEATURE_COLUMNS:
        grid[col] = design[col].to_numpy()

    # Causal demand history.  cnt is forward-filled first (so a missing hour
    # inherits the last observed demand), then every feature is shifted so that
    # it only ever contains information from earlier hours.
    demand = grid["cnt"].ffill()
    grid["lag1"] = demand.shift(1)
    grid["lag24"] = demand.shift(24)
    grid["lag168"] = demand.shift(168)
    grid["roll24"] = demand.shift(1).rolling(24, min_periods=24).mean()
    grid["roll168"] = demand.shift(1).rolling(168, min_periods=168).mean()

    grid["observed"] = grid["cnt"].notna()
    return grid


def prepare_sequences(window: int = 168) -> dict:
    """
    Build sliding windows of *window* hourly feature vectors for the recurrent
    and convolutional notebooks.

    Each sample ends at the prediction hour, so a window contains the current
    hour plus the immediately preceding ``window - 1`` hours.  The target is
    the observed cnt at the prediction hour.  Samples are restricted to the
    same frozen split rows and the same two-week warm-up as ``prepare_tabular``,
    which keeps all four methods on an identical sample set.

    The feature scaler is fitted on the training rows of the tabular design
    matrix, so it is the same transformation used by the other two methods.
    """
    if window < 1:
        raise ValueError("window must be at least 1")

    tabular = prepare_tabular()
    grid = build_hourly_grid()

    features = tabular["scaler"].transform(grid[FEATURE_COLUMNS])
    targets = grid[TARGET].to_numpy(dtype=np.float32)

    out = {"window": window, "grid": grid, "features": features,
           "targets": targets, "scaler": tabular["scaler"]}
    for name in ("train", "validation", "test"):
        positions = tabular["positions"][name]
        starts = positions - (window - 1)
        if (starts < WARMUP_HOURS).any():
            raise ValueError(
                f"window={window} starts before the history warm-up; "
                "use a window no longer than 168 hours"
            )
        idx = np.arange(window)
        X = features[starts[:, None] + idx].astype(np.float32)
        y = targets[positions].astype(np.float64)
        meta = grid.iloc[positions][["dteday", "hr", "weekday", "workingday",
                                     "season", "weathersit", "cnt"]].copy()
        meta["position"] = positions
        out[name] = {"X": X, "y": y, "meta": meta.reset_index(drop=True)}
    return out


def reshape_weekly(X: np.ndarray) -> np.ndarray:
    """
    Reshape (N, 168, F) windows into (N, F, 7, 24) weekly images.

    The 168 hours are read as seven consecutive days of 24 hours, with the
    seventh day containing the prediction hour at its last column.  The
    feature axis becomes the channel axis expected by Conv2d.
    """
    if X.shape[1] != 168:
        raise ValueError("reshape_weekly expects exactly 168 hourly steps")
    return X.reshape(X.shape[0], 7, 24, X.shape[2]).transpose(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "mean_error": float(np.mean(y_pred - y_true)),
    }


def peak_hours(train: pd.DataFrame, fraction: float = 0.25) -> list[int]:
    """Hours whose average training demand is in the top *fraction*."""
    profile = train.groupby("hr")["cnt"].mean().sort_values(ascending=False)
    n = max(1, int(round(len(profile) * fraction)))
    return sorted(int(h) for h in profile.head(n).index)


def high_demand_threshold(train: pd.DataFrame, quantile: float = 0.75) -> float:
    return float(train["cnt"].quantile(quantile))


def scenario_metrics(meta: pd.DataFrame, y_true, y_pred,
                     train: pd.DataFrame) -> dict:
    """
    Break the test error down by the scenarios the report discusses.
    ``meta`` must contain hr, weekday and workingday for the evaluated rows.
    """
    meta = meta.reset_index(drop=True)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_pred - y_true)

    peak = set(peak_hours(train))
    threshold = high_demand_threshold(train)
    is_peak = meta["hr"].isin(peak).to_numpy()
    is_weekend = (meta["workingday"] == 0).to_numpy()
    is_high = y_true >= threshold

    result = {}
    for name, mask in {
        "peak_hours": is_peak,
        "off_peak_hours": ~is_peak,
        "weekday": ~is_weekend,
        "weekend_or_holiday": is_weekend,
        "high_demand": is_high,
        "lower_demand": ~is_high,
    }.items():
        result[name] = {
            "n": int(mask.sum()),
            "MAE": float(abs_err[mask].mean()) if mask.any() else None,
            "RMSE": float(np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2)))
            if mask.any() else None,
        }

    by_hour = (
        pd.DataFrame({"hr": meta["hr"].to_numpy(), "abs_err": abs_err})
        .groupby("hr")["abs_err"].agg(["mean", "count"])
        .rename(columns={"mean": "MAE", "count": "n"})
    )
    result["by_hour"] = {
        int(h): {"MAE": float(row["MAE"]), "n": int(row["n"])}
        for h, row in by_hour.iterrows()
    }
    return result


def measure_inference(model_predict, X, repeats: int = 5) -> dict:
    """Measure single-row and batch inference latency on this machine."""
    X = np.asarray(X)
    model_predict(X[:2])  # warm-up
    t0 = time.perf_counter()
    for _ in range(repeats):
        model_predict(X)
    batch = (time.perf_counter() - t0) / repeats

    t0 = time.perf_counter()
    for i in range(min(200, len(X))):
        model_predict(X[i:i + 1])
    single = (time.perf_counter() - t0) / min(200, len(X))
    return {
        "batch_seconds": float(batch),
        "batch_rows": int(len(X)),
        "single_row_ms": float(single * 1000),
    }


def save_json(obj, name: str, directory: Path | None = None) -> Path:
    directory = directory or RESULTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
    return path


def load_json(name: str, directory: Path | None = None):
    directory = directory or RESULTS_DIR
    with open(directory / name, encoding="utf-8") as fh:
        return json.load(fh)


def setup_plot_style() -> None:
    """Consistent, readable matplotlib defaults for every notebook."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 160,
        "figure.figsize": (7.2, 4.2),
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.frameon": False,
        "font.size": 9,
    })


def save_figure(fig, name: str, directory: Path | None = None) -> Path:
    directory = directory or FIGURES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    fig.savefig(path, bbox_inches="tight")
    return path


def fmt_metrics(metrics: dict) -> str:
    return (f"MAE {metrics['MAE']:.2f} | RMSE {metrics['RMSE']:.2f} | "
            f"R2 {metrics['R2']:.4f}")
