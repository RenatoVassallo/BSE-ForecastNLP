from __future__ import annotations

from typing import Mapping, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_curve


def optimal_f1_threshold(y_true, y_score) -> float:
    """Return the probability threshold that maximizes the F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        return 0.5

    precision = precision[:-1]
    recall = recall[:-1]
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])


def add_binary_predictions(df: pd.DataFrame, score_columns: Mapping[str, str]) -> tuple[pd.DataFrame, dict[str, float]]:
    """Add binary prediction columns using F1-optimal thresholds."""
    out = df.copy()
    thresholds: dict[str, float] = {}

    for score_col, pred_col in score_columns.items():
        threshold = optimal_f1_threshold(out["is_financial"], out[score_col])
        thresholds[pred_col] = threshold
        out[pred_col] = (out[score_col] >= threshold).astype(int)

    return out, thresholds


def build_bci_series(
    df: pd.DataFrame,
    *,
    relevance_pred_col: str,
    positive_col: str,
    neutral_col: str,
    negative_col: str,
    weights: Optional[Mapping[str, float]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build article-level sentiment features and a monthly BCI series."""
    weights = weights or {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    score_cols = [positive_col, neutral_col, negative_col]
    out["max_sentiment"] = out[score_cols].idxmax(axis=1).str.lower()
    confidence = out[score_cols].max(axis=1)
    out["sentiment_score"] = out["max_sentiment"].map(weights) * confidence

    financial_df = out.loc[out[relevance_pred_col] == 1].copy()
    financial_df["date"] = financial_df["date"].dt.to_period("M")

    monthly_bci = financial_df.groupby("date")["sentiment_score"].mean()
    all_months = pd.period_range(financial_df["date"].min(), financial_df["date"].max(), freq="M")
    monthly_bci = monthly_bci.reindex(all_months, fill_value=0)
    monthly_bci = (monthly_bci - monthly_bci.mean()) / monthly_bci.std()
    monthly_bci = monthly_bci.reset_index().rename(columns={"index": "date", "sentiment_score": "BCI"})
    return out, monthly_bci


def plot_bci_and_investment(df: pd.DataFrame, *, bci_col: str = "BCI", target_col: str = "investment_growth") -> None:
    """Plot the monthly BCI against private investment growth."""
    plot_df = df.copy()
    if pd.api.types.is_period_dtype(plot_df["date"]):
        plot_df["date"] = plot_df["date"].dt.to_timestamp()
    else:
        plot_df["date"] = pd.to_datetime(plot_df["date"])
    plot_df = plot_df.set_index("date")

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(plot_df.index, plot_df[target_col], color="#2f6db3", linewidth=2, label="Investment growth")
    ax1.set_ylabel("Investment growth", color="#2f6db3")
    ax1.tick_params(axis="y", labelcolor="#2f6db3")
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(plot_df.index, plot_df[bci_col], color="#d95f02", linewidth=2, label="Business confidence index")
    ax2.set_ylabel("Business confidence index", color="#d95f02")
    ax2.tick_params(axis="y", labelcolor="#d95f02")

    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title("Business confidence and private investment")
    plt.tight_layout()
    plt.show()


def dynamic_correlation_plot(df: pd.DataFrame, fixed_var: str, dynamic_var: str, max_lag: int = 8) -> tuple[int, float]:
    """Plot contemporaneous and lead-lag correlations."""
    correlations = []
    lags = list(range(max_lag, -(max_lag + 1), -1))

    for lag in lags:
        shifted = df[dynamic_var].shift(lag)
        correlations.append(df[fixed_var].corr(shifted))

    best_idx = int(np.argmax(np.abs(correlations)))
    best_lag = lags[best_idx]
    best_corr = float(correlations[best_idx])

    plt.figure(figsize=(7, 4))
    plt.plot(lags, correlations, marker="o", color="#2f6db3")
    plt.axvline(best_lag, color="#d95f02", linestyle="--", label=f"Best lag = {best_lag}")
    plt.title("Dynamic correlation")
    plt.xlabel("Lag (months)")
    plt.ylabel("Correlation")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if best_lag < 0:
        print(f"{dynamic_var} lags {fixed_var} by {abs(best_lag)} months. Correlation: {best_corr:.2f}")
    elif best_lag > 0:
        print(f"{dynamic_var} leads {fixed_var} by {best_lag} months. Correlation: {best_corr:.2f}")
    else:
        print(f"{dynamic_var} and {fixed_var} move together. Correlation: {best_corr:.2f}")

    return best_lag, best_corr


def build_manual_ar_design(
    df: pd.DataFrame,
    *,
    target_col: str,
    lags: int = 1,
    exog_col: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the design matrix used in the manual AR regression."""
    y = df[target_col].to_numpy()[lags:].reshape(-1, 1)
    x = np.array([df[target_col].shift(i).dropna().to_numpy() for i in range(1, lags + 1)]).T

    if exog_col:
        exog = df[exog_col].shift(1).dropna().to_numpy().reshape(-1, 1)
        if len(exog) != len(x):
            raise ValueError("Lagged exogenous regressor has incompatible length.")
        x = np.hstack([x, exog])

    x = np.hstack([np.ones((x.shape[0], 1)), x])
    return y, x


def rolling_autoregressive_forecast(
    df: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    target_col: str = "investment_growth",
    lags: int = 1,
    exog_col: Optional[str] = None,
) -> pd.DataFrame:
    """Run a recursive one-step-ahead AR forecast."""
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise TypeError("The 'date' column must be datetime-like.")

    work_df = df.sort_values("date").reset_index(drop=True).copy()
    predictions = []

    for forecast_period in pd.period_range(start=start_date, end=end_date, freq="M"):
        forecast_date = forecast_period.to_timestamp()
        train_df = work_df.loc[work_df["date"] <= forecast_date]

        if len(train_df) < lags + 1:
            continue

        try:
            y, x = build_manual_ar_design(train_df, target_col=target_col, lags=lags, exog_col=exog_col)
            beta = np.linalg.solve(x.T @ x, x.T @ y)
        except np.linalg.LinAlgError:
            continue

        latest_target = train_df[target_col].iloc[-1]
        x_forecast = np.array([1] + [latest_target] * lags, dtype=float)

        if exog_col:
            x_forecast = np.hstack([x_forecast, train_df[exog_col].iloc[-1]])

        forecast = x_forecast.reshape(1, -1) @ beta
        predictions.append(
            {
                "date": forecast_date + pd.offsets.MonthBegin(1),
                "forecast": float(forecast.item()),
            }
        )

    pred_df = pd.DataFrame(predictions)
    return work_df.merge(pred_df, on="date", how="left")


def plot_forecasts(df_actual: pd.DataFrame, df_forecast_1: pd.DataFrame, df_forecast_2: pd.DataFrame) -> None:
    """Plot actual values and two competing forecast paths."""
    plt.figure(figsize=(12, 6))
    plt.plot(df_actual["date"], df_actual["investment_growth"], label="Actual", color="black", linestyle="--", linewidth=2)
    plt.plot(df_forecast_1["date"], df_forecast_1["forecast"], label="AR(1)", color="#2f6db3", linewidth=2)
    plt.plot(df_forecast_2["date"], df_forecast_2["forecast"], label="AR(1) + BCI", color="#d95f02", linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("Investment growth")
    plt.title("Recursive forecasts")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def calculate_forecast_metrics(df_actual: pd.DataFrame, df_forecast: pd.DataFrame) -> dict[str, float]:
    """Compute RMSE and MAE for a forecast path."""
    merged = pd.merge(df_actual, df_forecast[["date", "forecast"]], on="date", how="left").dropna(subset=["forecast"])
    rmse = float(np.sqrt(mean_squared_error(merged["investment_growth"], merged["forecast"])))
    mae = float(mean_absolute_error(merged["investment_growth"], merged["forecast"]))
    return {"RMSE": rmse, "MAE": mae}


def print_forecast_metrics(metrics: Mapping[str, float], label: str) -> None:
    """Print a compact metrics summary."""
    print(f"{label}: RMSE = {metrics['RMSE']:.3f}, MAE = {metrics['MAE']:.3f}")
