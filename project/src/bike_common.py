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


# ---------------------------------------------------------------------------
# Notebook helpers
#
# The four method notebooks share the same training loop, diagnostics plots
# and evidence files.  Keeping that plumbing here means each notebook only
# shows what is specific to its method: the model, the tuning experiment and
# the result.  torch is imported lazily so the interface can import this
# module without it.
# ---------------------------------------------------------------------------
SEED = 42
META_COLUMNS = ["instant", "dteday", "hr", "weekday", "workingday",
                "season", "weathersit", "cnt"]


def train_regressor(model_fn, X_train, y_train, X_val, y_val, y_mean, y_std, *,
                    max_epochs=15, patience=4, lr=1e-3, batch_size=128,
                    clip=True, use_scheduler=True, tag="model", verbose=True):
    """
    Train a PyTorch regressor on the standardised target with early stopping.

    ``model_fn`` builds the network after the seed is set, so every run starts
    from the same weights.  The loop uses AdamW, smooth L1 loss, optional
    gradient clipping and an optional ReduceLROnPlateau scheduler; it stops
    when validation MAE (in rentals) has not improved by 0.05 for ``patience``
    epochs and restores the best epoch's weights.
    """
    import copy
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(SEED)
    model = model_fn()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
                 if use_scheduler else None)
    loss_fn = nn.SmoothL1Loss(beta=0.5)

    Xtr = torch.tensor(np.asarray(X_train, dtype=np.float32))
    ytr = torch.tensor((np.asarray(y_train) - y_mean) / y_std, dtype=torch.float32).view(-1, 1)
    Xva = torch.tensor(np.asarray(X_val, dtype=np.float32))
    yva = torch.tensor((np.asarray(y_val) - y_mean) / y_std, dtype=torch.float32).view(-1, 1)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))

    history, best_mae, best_epoch, bad = [], float("inf"), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    t0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * len(xb)
        train_loss = running / len(Xtr)

        model.eval()
        with torch.no_grad():
            val_scaled = model(Xva)
            val_loss = loss_fn(val_scaled, yva).item()
        val_mae = regression_metrics(y_val, val_scaled.numpy().ravel() * y_std + y_mean)["MAE"]
        if scheduler is not None:
            scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_MAE": val_mae})
        if verbose:
            print(f"{tag} epoch {epoch:02d}  train {train_loss:.4f}  "
                  f"val {val_loss:.4f}  val MAE {val_mae:.2f}", flush=True)

        if val_mae < best_mae - 0.05:
            best_mae, best_epoch, bad = val_mae, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                print(f"{tag}: early stop at epoch {epoch} "
                      f"(best epoch {best_epoch}, MAE {best_mae:.2f})")
                break

    model.load_state_dict(best_state)
    model.eval()
    return {"model": model, "history": pd.DataFrame(history), "best_epoch": best_epoch,
            "best_val_MAE": best_mae, "train_seconds": time.perf_counter() - t0}


def predict_regressor(model, X, y_mean, y_std):
    """Predict rentals per hour with a trained PyTorch regressor."""
    import torch

    model.eval()
    with torch.no_grad():
        scaled = model(torch.tensor(np.asarray(X, dtype=np.float32))).numpy().ravel()
    return scaled * y_std + y_mean


def evaluate(y_val, val_pred, y_test, test_pred, meta_val, meta_test, train) -> dict:
    """Overall and scenario metrics on validation and test, printed and returned."""
    out = {
        "validation": regression_metrics(y_val, val_pred),
        "test": regression_metrics(y_test, test_pred),
        "validation_scenarios": scenario_metrics(meta_val, y_val, val_pred, train),
        "test_scenarios": scenario_metrics(meta_test, y_test, test_pred, train),
        "peak_hours": peak_hours(train),
        "high_demand_threshold": high_demand_threshold(train),
    }
    print("Validation:", fmt_metrics(out["validation"]))
    print("Test:      ", fmt_metrics(out["test"]))
    return out


def scenario_table(ev: dict) -> pd.DataFrame:
    """Scenario MAE table (validation and test) from :func:`evaluate`."""
    val, test = ev["validation_scenarios"], ev["test_scenarios"]
    keys = [k for k in val if k != "by_hour"]
    return pd.DataFrame({
        "validation n": {k: val[k]["n"] for k in keys},
        "validation MAE": {k: val[k]["MAE"] for k in keys},
        "test n": {k: test[k]["n"] for k in keys},
        "test MAE": {k: test[k]["MAE"] for k in keys},
    }).round(2)


def plot_training_curves(history: pd.DataFrame, best_epoch: int, title: str, name: str):
    """Loss curves and validation MAE per epoch, with the restored epoch marked."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))
    axes[0].plot(history["epoch"], history["train_loss"], marker="o", ms=3, label="training")
    axes[0].plot(history["epoch"], history["val_loss"], marker="s", ms=3, label="validation")
    axes[0].set(title=f"{title} loss curves", xlabel="Epoch",
                ylabel="Smooth L1 loss (scaled target)")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["val_MAE"], marker="o", ms=3, color="#c00000")
    axes[1].axvline(best_epoch, color="black", ls="--", lw=1, label=f"best epoch {best_epoch}")
    axes[1].set(title="Validation MAE during training", xlabel="Epoch", ylabel="MAE (rentals)")
    axes[1].legend()
    plt.tight_layout()
    save_figure(fig, name)
    plt.show()


def plot_diagnostics(y_true, y_pred, hours, title: str, prefix: str) -> None:
    """Predicted-vs-observed hexbin, residual histogram and MAE by hour (test)."""
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    hb = ax.hexbin(y_true, y_pred, gridsize=45, cmap="Blues", mincnt=1)
    lims = [0, max(y_true.max(), y_pred.max()) * 1.02]
    ax.plot(lims, lims, color="#c00000", ls="--", lw=1, label="perfect prediction")
    ax.set(xlim=lims, ylim=lims, xlabel="Observed rentals", ylabel="Predicted rentals",
           title=f"{title} — test period")
    ax.legend(loc="upper left")
    fig.colorbar(hb, ax=ax, label="hours")
    save_figure(fig, f"fig_{prefix}_pred_vs_actual.png")
    plt.show()

    residual = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    axes[0].hist(residual, bins=55, color="#1f4e79", alpha=0.85)
    axes[0].axvline(0, color="#c00000", ls="--")
    axes[0].set(title="Test residuals (predicted − observed)",
                xlabel="Error in rentals", ylabel="Hours")
    by_hour = pd.Series(np.abs(residual)).groupby(np.asarray(hours)).mean()
    axes[1].bar(by_hour.index, by_hour.values, color="#5b9bd5")
    axes[1].set(title="Mean absolute error by hour (test)",
                xlabel="Hour of day", ylabel="MAE (rentals)")
    axes[1].set_xticks(range(0, 24, 3))
    plt.tight_layout()
    save_figure(fig, f"fig_{prefix}_residuals.png")
    plt.show()
    print(f"Residual mean {residual.mean():.2f} | standard deviation {residual.std():.2f}")


def worst_errors(meta: pd.DataFrame, y_pred, n: int = 12) -> pd.DataFrame:
    """The *n* test hours with the largest absolute error, with their context."""
    frame = meta.reset_index(drop=True).copy()
    frame["prediction"] = np.asarray(y_pred)
    frame["error"] = frame["prediction"] - frame["cnt"]
    cols = [c for c in ["dteday", "hr", "weekday", "workingday", "weathersit",
                        "temp", "hum", "cnt", "prediction", "error"] if c in frame]
    table = frame.loc[frame["error"].abs().sort_values(ascending=False).index[:n], cols]
    numeric = [c for c in table.columns if pd.api.types.is_numeric_dtype(table[c])]
    return table.round({c: 1 for c in numeric})


def save_predictions(meta: pd.DataFrame, y_pred, name: str) -> None:
    """Write one prediction CSV keyed by ``instant`` for the comparison notebook."""
    frame = meta.reset_index(drop=True)[[c for c in META_COLUMNS if c in meta]].copy()
    frame["prediction"] = np.asarray(y_pred)
    frame.to_csv(RESULTS_DIR / name, index=False)


def file_mb(*paths) -> float:
    return sum(Path(p).stat().st_size for p in paths) / (1024 * 1024)
