from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TIMEFRAME_TO_FREQ = {
    "M1": "t",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "h",
    "H4": "4h",
    "D1": "d",
}


@dataclass(frozen=True)
class CustomSplitInfo:
    num_train: int
    num_val: int
    num_test: int
    test_border1: int
    test_border2: int


def normalize_custom_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'date' column.")
    if target not in df.columns:
        raise ValueError(f"CSV must contain the target column '{target}'.")

    ordered = df.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], errors="raise")
    feature_cols = [col for col in ordered.columns if col not in {"date", target}]
    ordered = ordered[["date", *feature_cols, target]]

    if ordered.isna().any().any():
        ordered = ordered.dropna().reset_index(drop=True)

    return ordered


def infer_channel_size(df: pd.DataFrame, features: str) -> int:
    if features == "S":
        return 1
    return len(df.columns) - 1


def get_freq_for_timeframe(timeframe: str) -> str:
    upper = timeframe.upper()
    if upper not in TIMEFRAME_TO_FREQ:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. "
            f"Supported values: {', '.join(sorted(TIMEFRAME_TO_FREQ))}."
        )
    return TIMEFRAME_TO_FREQ[upper]


def compute_custom_split(n_rows: int, seq_len: int) -> CustomSplitInfo:
    num_train = int(n_rows * 0.7)
    num_test = int(n_rows * 0.2)
    num_val = n_rows - num_train - num_test

    test_border1 = n_rows - num_test - seq_len
    test_border2 = n_rows
    if test_border1 < 0:
        raise ValueError(
            f"Dataset is too short for seq_len={seq_len}. "
            f"Need more than {seq_len} rows, got {n_rows}."
        )

    return CustomSplitInfo(
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
        test_border1=test_border1,
        test_border2=test_border2,
    )


def build_setting_name(
    *,
    task_name: str,
    model_id: str,
    model: str,
    data: str,
    features: str,
    seq_len: int,
    label_len: int,
    pred_len: int,
    d_model: int,
    n_heads: int,
    e_layers: int,
    d_layers: int,
    d_ff: int,
    expand: int,
    d_conv: int,
    factor: int,
    embed: str,
    distil: bool,
    des: str,
    itr_index: int,
) -> str:
    return (
        f"{task_name}_{model_id}_{model}_{data}_ft{features}"
        f"_sl{seq_len}_ll{label_len}_pl{pred_len}_dm{d_model}"
        f"_nh{n_heads}_el{e_layers}_dl{d_layers}_df{d_ff}"
        f"_expand{expand}_dc{d_conv}_fc{factor}_eb{embed}_dt{distil}"
        f"_{des}_{itr_index}"
    )


def sanitize_model_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)
    return cleaned.strip("_") or "quant_run"


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _safe_float(value: float | np.floating[Any] | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isfinite(value):
        return value
    return None


def evaluate_backtest(
    *,
    csv_path: Path,
    result_dir: Path,
    target: str,
    seq_len: int,
    pred_step: int = 0,
    target_dim: int = -1,
    threshold_bps: float = 0.0,
    cost_bps: float = 1.5,
) -> dict[str, Any]:
    df = normalize_custom_frame(pd.read_csv(csv_path), target=target)
    split = compute_custom_split(len(df), seq_len=seq_len)

    pred = np.load(result_dir / "pred.npy")
    true = np.load(result_dir / "true.npy")

    if pred.ndim != 3 or true.ndim != 3:
        raise ValueError("Expected pred.npy and true.npy to have shape [samples, pred_len, channels].")
    if pred.shape != true.shape:
        raise ValueError(f"Prediction/label shape mismatch: {pred.shape} vs {true.shape}.")
    if pred_step < 0 or pred_step >= pred.shape[1]:
        raise ValueError(f"pred_step={pred_step} is out of range for pred_len={pred.shape[1]}.")

    expected_windows = split.num_test - pred.shape[1] + 1
    if pred.shape[0] != expected_windows:
        raise ValueError(
            "Prediction window count does not match Dataset_Custom split logic. "
            f"Expected {expected_windows}, got {pred.shape[0]}."
        )

    pred_prices = pred[:, pred_step, target_dim]
    true_prices = true[:, pred_step, target_dim]

    forecast_indices = split.test_border1 + seq_len + pred_step + np.arange(pred.shape[0])
    prev_indices = forecast_indices - 1

    prev_close = df.iloc[prev_indices][target].to_numpy(dtype=float)
    forecast_time = df.iloc[forecast_indices]["date"].to_numpy()

    predicted_return = (pred_prices - prev_close) / prev_close
    realized_return = (true_prices - prev_close) / prev_close

    threshold = threshold_bps / 10000.0
    signal = np.where(predicted_return > threshold, 1, np.where(predicted_return < -threshold, -1, 0))

    gross_return = signal * realized_return
    # cost_bps is defined as the round-trip trading cost in basis points.
    cost_per_trade = cost_bps / 10000.0
    net_return = gross_return - np.where(signal != 0, cost_per_trade, 0.0)

    trade_mask = signal != 0
    trade_returns = net_return[trade_mask]
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]

    equity = np.cumprod(1.0 + net_return)
    equity_curve = np.concatenate(([1.0], equity))
    running_peak = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve / running_peak - 1.0

    avg_win = float(wins.mean()) if wins.size else None
    avg_loss = float(np.abs(losses.mean())) if losses.size else None

    summary = {
        "csv_path": str(csv_path),
        "result_dir": str(result_dir),
        "rows": int(len(df)),
        "test_rows": int(split.num_test),
        "prediction_windows": int(pred.shape[0]),
        "pred_len": int(pred.shape[1]),
        "pred_step": int(pred_step),
        "trade_count": int(trade_mask.sum()),
        "coverage": _safe_float(trade_mask.mean()),
        "win_rate": _safe_float((trade_returns > 0).mean() if trade_returns.size else None),
        "avg_win": _safe_float(avg_win),
        "avg_loss": _safe_float(avg_loss),
        "win_loss_ratio": _safe_float(avg_win / avg_loss if avg_win is not None and avg_loss else None),
        "profit_factor": _safe_float(wins.sum() / np.abs(losses.sum()) if losses.size else None),
        "cumulative_return": _safe_float(equity[-1] - 1.0 if equity.size else 0.0),
        "max_drawdown": _safe_float(abs(drawdown.min())),
        "mean_trade_return": _safe_float(trade_returns.mean() if trade_returns.size else None),
        "pred_direction_acc": _safe_float(
            (np.sign(predicted_return[trade_mask]) == np.sign(realized_return[trade_mask])).mean()
            if trade_returns.size
            else None
        ),
        "threshold_bps": float(threshold_bps),
        "cost_bps": float(cost_bps),
    }

    trade_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(forecast_time),
            "prev_close": prev_close,
            "pred_close": pred_prices,
            "true_close": true_prices,
            "predicted_return": predicted_return,
            "realized_return": realized_return,
            "signal": signal,
            "gross_return": gross_return,
            "net_return": net_return,
            "equity": equity,
        }
    )

    trade_frame.to_csv(result_dir / "backtest_trades.csv", index=False)
    save_json(summary, result_dir / "backtest_summary.json")

    return summary
