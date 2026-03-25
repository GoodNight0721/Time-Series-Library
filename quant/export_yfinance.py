from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from quant.common import TIMEFRAME_TO_FREQ, save_json

YF_INTERVAL_MAP = {
    "M15": ("15m", None),
    "H1": ("60m", None),
    "H4": ("60m", "4h"),
    "D1": ("1d", None),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Yahoo Finance bars into TSLib custom CSV format.")
    parser.add_argument("--ticker", required=True, help="Yahoo ticker, for example EURUSD=X, GC=F, NQ=F.")
    parser.add_argument("--symbol", required=True, help="Output symbol name, for example EURUSD or XAUUSD.")
    parser.add_argument("--timeframe", required=True, choices=sorted(YF_INTERVAL_MAP), help="One of M15, H1, H4, D1.")
    parser.add_argument("--period", default="730d", help="Yahoo period, for example 60d, 730d, max.")
    parser.add_argument("--output-dir", default="dataset/mt5_fallback", help="Directory used to store the exported CSV.")
    parser.add_argument("--target", default="close", help="Prediction target column.")
    return parser.parse_args()


def download_bars(ticker: str, timeframe: str, period: str) -> pd.DataFrame:
    interval, resample_rule = YF_INTERVAL_MAP[timeframe]
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise SystemExit(f"No bars returned for ticker={ticker}, timeframe={timeframe}, period={period}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "tick_volume",
        }
    )
    df.index = pd.to_datetime(df.index).tz_localize(None)

    if resample_rule:
        agg_map = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "tick_volume": "sum",
        }
        if "adj_close" in df.columns:
            agg_map["adj_close"] = "last"
        df = df.resample(resample_rule).agg(agg_map).dropna()

    return df.reset_index().rename(columns={"Datetime": "date", "Date": "date"})


def build_feature_frame(raw_df: pd.DataFrame, target: str) -> pd.DataFrame:
    df = raw_df.copy()
    if "date" not in df.columns:
        raise ValueError("Downloaded data does not contain a date column.")

    if "tick_volume" not in df.columns:
        df["tick_volume"] = 0.0

    df["spread"] = 0.0
    df["real_volume"] = df["tick_volume"]
    df["range"] = df["high"] - df["low"]
    df["body"] = df["close"] - df["open"]
    df["return_1"] = df["close"].pct_change().fillna(0.0)
    df["log_return_1"] = np.log(df["close"]).diff().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["hl_ratio"] = ((df["high"] - df["low"]) / df["close"]).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    if target not in df.columns:
        raise ValueError(f"Target '{target}' is not available in downloaded data.")

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
    dataset = df[["date", *feature_cols, target]].copy()
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return dataset


def main() -> None:
    args = parse_args()
    raw = download_bars(args.ticker, args.timeframe, args.period)
    dataset = build_feature_frame(raw, target=args.target)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.symbol}_{args.timeframe}.csv"
    dataset.to_csv(csv_path, index=False)

    summary = {
        "symbol": args.symbol,
        "ticker": args.ticker,
        "timeframe": args.timeframe,
        "freq": TIMEFRAME_TO_FREQ[args.timeframe],
        "rows": int(len(dataset)),
        "start": pd.to_datetime(dataset["date"]).min().isoformat(),
        "end": pd.to_datetime(dataset["date"]).max().isoformat(),
        "target": args.target,
        "feature_count": int(len(dataset.columns) - 1),
        "feature_columns": [col for col in dataset.columns if col != "date"],
    }
    save_json(summary, output_dir / f"{args.symbol}_{args.timeframe}.meta.json")

    print(f"Saved {len(dataset)} rows to {csv_path}")
    print(f"Suggested TSLib freq: {summary['freq']}")


if __name__ == "__main__":
    main()
