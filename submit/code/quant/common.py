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


def _trend_filter_signal(
    *,
    predicted_return: np.ndarray,
    prev_close: np.ndarray,
    ma_prev: np.ndarray | None,
    threshold: float,
    allow_short: bool,
) -> np.ndarray:
    long_mask = predicted_return > threshold
    short_mask = predicted_return < -threshold

    if ma_prev is not None:
        valid_ma = np.isfinite(ma_prev)
        long_mask &= valid_ma & (prev_close > ma_prev)
        short_mask &= valid_ma & (prev_close < ma_prev)

    if not allow_short:
        short_mask = np.zeros_like(short_mask, dtype=bool)

    return np.where(long_mask, 1, np.where(short_mask, -1, 0))


def _position_trend_ok(position: int, price: float, ma_value: float | None) -> bool:
    if position == 0 or ma_value is None or not math.isfinite(ma_value):
        return True
    if position > 0:
        return price > ma_value
    return price < ma_value


def _finalize_trade(
    *,
    trades: list[dict[str, Any]],
    entry_time: pd.Timestamp | None,
    exit_time: pd.Timestamp,
    entry_price: float | None,
    exit_price: float,
    direction: int,
    holding_bars: int,
    trade_return: float,
) -> None:
    if entry_time is None or entry_price is None or direction == 0:
        return
    trades.append(
        {
            "entry_time": entry_time,
            "exit_time": exit_time,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "holding_bars": int(holding_bars),
            "trade_return": float(trade_return),
        }
    )


def _evaluate_stateful_backtest(
    *,
    csv_path: Path,
    result_dir: Path,
    threshold_bps: float,
    cost_bps: float,
    predicted_return: np.ndarray,
    realized_return: np.ndarray,
    prev_close: np.ndarray,
    true_prices: np.ndarray,
    forecast_time: np.ndarray,
    decision_time: np.ndarray,
    ma_prev: np.ndarray | None,
    min_hold_bars: int,
    hold_until_opposite: bool,
    stop_loss_bps: float,
    cooldown_bars: int,
    allow_short: bool,
    write_outputs: bool,
) -> dict[str, Any]:
    threshold = threshold_bps / 10000.0
    stop_loss = stop_loss_bps / 10000.0 if stop_loss_bps > 0 else None
    half_turn_cost = cost_bps / 20000.0

    raw_signal = _trend_filter_signal(
        predicted_return=predicted_return,
        prev_close=prev_close,
        ma_prev=ma_prev,
        threshold=threshold,
        allow_short=allow_short,
    )

    n = len(raw_signal)
    position_arr = np.zeros(n, dtype=int)
    gross_arr = np.zeros(n, dtype=float)
    net_arr = np.zeros(n, dtype=float)
    cost_arr = np.zeros(n, dtype=float)
    turnover_arr = np.zeros(n, dtype=float)

    position = 0
    bars_held = 0
    entry_price: float | None = None
    entry_time: pd.Timestamp | None = None
    entry_bar_index: int | None = None
    active_trade_return = 0.0
    force_flat_next = False
    cooldown_remaining = 0
    trades: list[dict[str, Any]] = []

    for i in range(n):
        current_raw = int(raw_signal[i])
        ma_value = None if ma_prev is None else float(ma_prev[i])
        current_prev_close = float(prev_close[i])
        current_close = float(true_prices[i])
        current_decision_time = pd.Timestamp(decision_time[i])
        current_forecast_time = pd.Timestamp(forecast_time[i])

        if position == 0 and cooldown_remaining > 0:
            current_raw = 0

        desired_position = current_raw if position == 0 else position

        if position != 0:
            if force_flat_next:
                desired_position = 0
            else:
                if bars_held < min_hold_bars:
                    desired_position = position
                elif not _position_trend_ok(position, current_prev_close, ma_value):
                    desired_position = 0 if current_raw == 0 else current_raw
                elif hold_until_opposite:
                    if current_raw == -position:
                        desired_position = current_raw
                    elif current_raw == 0:
                        desired_position = position
                    else:
                        desired_position = current_raw
                else:
                    desired_position = current_raw

        position_change = desired_position - position
        turnover = abs(position_change)
        turnover_arr[i] = turnover
        transition_cost = 0.0

        if position != 0 and desired_position != position:
            transition_cost += half_turn_cost
            active_trade_return -= half_turn_cost
            _finalize_trade(
                trades=trades,
                entry_time=entry_time,
                exit_time=current_decision_time,
                entry_price=entry_price,
                exit_price=current_prev_close,
                direction=position,
                holding_bars=bars_held,
                trade_return=active_trade_return,
            )
            entry_price = None
            entry_time = None
            entry_bar_index = None
            active_trade_return = 0.0
            bars_held = 0
            force_flat_next = False
            if desired_position == 0:
                cooldown_remaining = cooldown_bars

        if desired_position != 0 and desired_position != position:
            transition_cost += half_turn_cost
            entry_price = current_prev_close
            entry_time = current_decision_time
            entry_bar_index = i
            active_trade_return = -half_turn_cost
            bars_held = 0
            cooldown_remaining = 0

        gross_return = desired_position * float(realized_return[i])
        net_return = gross_return - transition_cost

        if desired_position != 0:
            active_trade_return += gross_return
            bars_held += 1
            if entry_price is not None and stop_loss is not None:
                trade_mark_return = desired_position * ((current_close / entry_price) - 1.0)
                force_flat_next = trade_mark_return <= -stop_loss
            else:
                force_flat_next = False
        else:
            if cooldown_remaining > 0:
                cooldown_remaining -= 1

        position = desired_position
        position_arr[i] = position
        gross_arr[i] = gross_return
        net_arr[i] = net_return
        cost_arr[i] = transition_cost

    if position != 0:
        final_close_cost = half_turn_cost
        net_arr[-1] -= final_close_cost
        cost_arr[-1] += final_close_cost
        active_trade_return -= final_close_cost
        _finalize_trade(
            trades=trades,
            entry_time=entry_time,
            exit_time=pd.Timestamp(forecast_time[-1]),
            entry_price=entry_price,
            exit_price=float(true_prices[-1]),
            direction=position,
            holding_bars=bars_held,
            trade_return=active_trade_return,
        )

    equity = np.cumprod(1.0 + net_arr)
    equity_curve = np.concatenate(([1.0], equity))
    running_peak = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve / running_peak - 1.0

    trade_returns = np.array([trade["trade_return"] for trade in trades], dtype=float)
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]

    avg_win = float(wins.mean()) if wins.size else None
    avg_loss = float(np.abs(losses.mean())) if losses.size else None
    active_mask = position_arr != 0

    summary = {
        "csv_path": str(csv_path),
        "result_dir": str(result_dir),
        "rows": int(len(prev_close) + 1),
        "test_rows": int(len(prev_close)),
        "prediction_windows": int(len(predicted_return)),
        "pred_len": 1,
        "pred_step": 0,
        "trade_count": int(len(trades)),
        "coverage": _safe_float(active_mask.mean()),
        "win_rate": _safe_float((trade_returns > 0).mean() if trade_returns.size else None),
        "avg_win": _safe_float(avg_win),
        "avg_loss": _safe_float(avg_loss),
        "win_loss_ratio": _safe_float(avg_win / avg_loss if avg_win is not None and avg_loss else None),
        "profit_factor": _safe_float(wins.sum() / np.abs(losses.sum()) if losses.size else None),
        "cumulative_return": _safe_float(equity[-1] - 1.0 if equity.size else 0.0),
        "max_drawdown": _safe_float(abs(drawdown.min())),
        "mean_trade_return": _safe_float(trade_returns.mean() if trade_returns.size else None),
        "pred_direction_acc": _safe_float(
            (np.sign(position_arr[active_mask]) == np.sign(realized_return[active_mask])).mean()
            if active_mask.any()
            else None
        ),
        "threshold_bps": float(threshold_bps),
        "cost_bps": float(cost_bps),
        "strategy_mode": "stateful",
        "ma_window": int(0 if ma_prev is None else 1),  # overwritten by caller
        "min_hold_bars": int(min_hold_bars),
        "hold_until_opposite": bool(hold_until_opposite),
        "stop_loss_bps": float(stop_loss_bps),
        "cooldown_bars": int(cooldown_bars),
        "allow_short": bool(allow_short),
    }

    bar_frame = pd.DataFrame(
        {
            "decision_time": pd.to_datetime(decision_time),
            "date": pd.to_datetime(forecast_time),
            "prev_close": prev_close,
            "true_close": true_prices,
            "predicted_return": predicted_return,
            "realized_return": realized_return,
            "raw_signal": raw_signal,
            "position": position_arr,
            "turnover": turnover_arr,
            "trade_cost": cost_arr,
            "gross_return": gross_arr,
            "net_return": net_arr,
            "equity": equity,
        }
    )
    if ma_prev is not None:
        bar_frame["ma_prev"] = ma_prev

    trade_frame = pd.DataFrame(trades)

    if write_outputs:
        bar_frame.to_csv(result_dir / "backtest_bars.csv", index=False)
        trade_frame.to_csv(result_dir / "backtest_trades.csv", index=False)
        save_json(summary, result_dir / "backtest_summary.json")

    return summary


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
    strategy_mode: str = "one_bar",
    ma_window: int = 0,
    min_hold_bars: int = 1,
    hold_until_opposite: bool = False,
    stop_loss_bps: float = 0.0,
    cooldown_bars: int = 0,
    allow_short: bool = True,
    write_outputs: bool = True,
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
    decision_time = df.iloc[prev_indices]["date"].to_numpy()

    predicted_return = (pred_prices - prev_close) / prev_close
    realized_return = (true_prices - prev_close) / prev_close

    ma_prev = None
    if ma_window > 0:
        rolling_ma = df[target].rolling(window=ma_window, min_periods=ma_window).mean().shift(1)
        ma_prev = rolling_ma.iloc[prev_indices].to_numpy(dtype=float)

    if strategy_mode == "stateful":
        summary = _evaluate_stateful_backtest(
            csv_path=csv_path,
            result_dir=result_dir,
            threshold_bps=threshold_bps,
            cost_bps=cost_bps,
            predicted_return=predicted_return,
            realized_return=realized_return,
            prev_close=prev_close,
            true_prices=true_prices,
            forecast_time=forecast_time,
            decision_time=decision_time,
            ma_prev=ma_prev,
            min_hold_bars=min_hold_bars,
            hold_until_opposite=hold_until_opposite,
            stop_loss_bps=stop_loss_bps,
            cooldown_bars=cooldown_bars,
            allow_short=allow_short,
            write_outputs=write_outputs,
        )
        summary["rows"] = int(len(df))
        summary["test_rows"] = int(split.num_test)
        summary["prediction_windows"] = int(pred.shape[0])
        summary["pred_len"] = int(pred.shape[1])
        summary["pred_step"] = int(pred_step)
        summary["ma_window"] = int(ma_window)
        return summary

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
        "strategy_mode": "one_bar",
        "ma_window": int(ma_window),
        "min_hold_bars": int(min_hold_bars),
        "hold_until_opposite": bool(hold_until_opposite),
        "stop_loss_bps": float(stop_loss_bps),
        "cooldown_bars": int(cooldown_bars),
        "allow_short": bool(allow_short),
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

    if write_outputs:
        trade_frame.to_csv(result_dir / "backtest_bars.csv", index=False)
        trade_frame.to_csv(result_dir / "backtest_trades.csv", index=False)
        save_json(summary, result_dir / "backtest_summary.json")

    return summary
