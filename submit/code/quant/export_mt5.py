from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quant.common import get_freq_for_timeframe, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MT5 bars into TSLib custom CSV format.")
    parser.add_argument("--symbol", required=True, help="MT5 symbol, for example EURUSD or XAUUSD.")
    parser.add_argument("--timeframe", required=True, help="One of M1, M5, M15, M30, H1, H4, D1.")
    parser.add_argument("--output-dir", default="dataset/mt5", help="Directory used to store the exported CSV.")
    parser.add_argument("--start", help="UTC start time, for example 2024-01-01T00:00:00.")
    parser.add_argument("--end", help="UTC end time, for example 2025-12-31T23:59:59.")
    parser.add_argument("--bars", type=int, default=5000, help="Used when --start/--end are omitted.")
    parser.add_argument("--target", default="close", help="Prediction target column.")
    parser.add_argument("--terminal-path", help="Optional path to terminal64.exe.")
    parser.add_argument("--login", type=int, help="Optional MT5 account login.")
    parser.add_argument("--password", help="Optional MT5 account password.")
    parser.add_argument("--server", help="Optional MT5 server name.")
    return parser.parse_args()


def resolve_timeframe(mt5_module: object, timeframe: str) -> int:
    attr = f"TIMEFRAME_{timeframe.upper()}"
    if not hasattr(mt5_module, attr):
        raise ValueError(f"Unsupported timeframe '{timeframe}'.")
    return getattr(mt5_module, attr)


def parse_utc_timestamp(value: str) -> datetime:
    ts = datetime.fromisoformat(value)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def load_rates(args: argparse.Namespace) -> pd.DataFrame:
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise SystemExit(
            "MetaTrader5 is not installed. Run `pip install MetaTrader5` in the same environment first."
        ) from exc

    init_kwargs = {}
    if args.terminal_path:
        init_kwargs["path"] = args.terminal_path
    if args.login:
        init_kwargs["login"] = args.login
    if args.password:
        init_kwargs["password"] = args.password
    if args.server:
        init_kwargs["server"] = args.server

    if not mt5.initialize(**init_kwargs):
        raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")

    try:
        timeframe = resolve_timeframe(mt5, args.timeframe)
        if args.start and args.end:
            rates = mt5.copy_rates_range(
                args.symbol,
                timeframe,
                parse_utc_timestamp(args.start),
                parse_utc_timestamp(args.end),
            )
        else:
            rates = mt5.copy_rates_from_pos(args.symbol, timeframe, 0, args.bars)
    finally:
        mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise SystemExit("No MT5 bars were returned. Check the symbol, timeframe, and date range.")

    return pd.DataFrame(rates)


def build_feature_frame(raw_df: pd.DataFrame, target: str) -> pd.DataFrame:
    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)
    df["range"] = df["high"] - df["low"]
    df["body"] = df["close"] - df["open"]
    df["return_1"] = df["close"].pct_change().fillna(0.0)
    df["log_return_1"] = np.log(df["close"]).diff().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["hl_ratio"] = ((df["high"] - df["low"]) / df["close"]).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    if target not in df.columns:
        raise ValueError(f"Target '{target}' is not available in MT5 bar data.")

    feature_cols = [
        "open",
        "high",
        "low",
        "tick_volume",
        "spread",
        "real_volume",
        "range",
        "body",
        "return_1",
        "log_return_1",
        "hl_ratio",
    ]
    existing_feature_cols = [col for col in feature_cols if col in df.columns and col != target]

    dataset = df[["date", *existing_feature_cols, target]].copy()
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    numeric_cols = [col for col in dataset.columns if col != "date"]
    for col in numeric_cols:
        dataset[col] = dataset[col].astype(float)

    return dataset


def main() -> None:
    args = parse_args()
    raw = load_rates(args)
    dataset = build_feature_frame(raw, target=args.target)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.symbol}_{args.timeframe.upper()}.csv"
    dataset.to_csv(csv_path, index=False)

    summary = {
        "symbol": args.symbol,
        "timeframe": args.timeframe.upper(),
        "freq": get_freq_for_timeframe(args.timeframe),
        "rows": int(len(dataset)),
        "start": dataset["date"].min().isoformat(),
        "end": dataset["date"].max().isoformat(),
        "target": args.target,
        "feature_count": int(len(dataset.columns) - 1),
        "feature_columns": [col for col in dataset.columns if col != "date"],
    }
    save_json(summary, output_dir / f"{args.symbol}_{args.timeframe.upper()}.meta.json")

    print(f"Saved {len(dataset)} rows to {csv_path}")
    print(f"Suggested TSLib freq: {summary['freq']}")


if __name__ == "__main__":
    main()
