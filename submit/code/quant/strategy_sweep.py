from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

from quant.common import evaluate_backtest


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep stateful strategy parameters over existing prediction folders.")
    parser.add_argument("--summary-csv", required=True, help="CSV produced by quant.run_grid with csv_path/result_dir columns.")
    parser.add_argument("--out", required=True, help="Where to write the sweep results.")
    parser.add_argument("--thresholds", default="8,10,12,15,20")
    parser.add_argument("--ma-windows", default="0,20,50")
    parser.add_argument("--min-holds", default="1,2,3")
    parser.add_argument("--stop-losses", default="0,20,40")
    parser.add_argument("--cooldowns", default="0,1,2")
    parser.add_argument("--hold-until-opposite", default="false,true")
    parser.add_argument("--long-only-options", default="false,true")
    parser.add_argument("--top-k", type=int, default=0, help="Keep only top-k rows per symbol/timeframe/model by cumulative return.")
    return parser.parse_args()


def _parse_bool_options(raw: str) -> list[bool]:
    mapping = {"true": True, "false": False}
    values = []
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item not in mapping:
            raise ValueError(f"Unsupported boolean option '{item}'. Use true/false.")
        values.append(mapping[item])
    return values


def main() -> None:
    args = parse_args()
    summary_df = pd.read_csv(args.summary_csv)
    if "status" in summary_df.columns:
        summary_df = summary_df[summary_df["status"] == "ok"].copy()

    thresholds = _parse_float_list(args.thresholds)
    ma_windows = _parse_int_list(args.ma_windows)
    min_holds = _parse_int_list(args.min_holds)
    stop_losses = _parse_float_list(args.stop_losses)
    cooldowns = _parse_int_list(args.cooldowns)
    hold_options = _parse_bool_options(args.hold_until_opposite)
    long_only_options = _parse_bool_options(args.long_only_options)

    rows: list[dict[str, object]] = []
    for row in summary_df.itertuples(index=False):
        csv_path = Path(row.csv_path)
        result_dir = Path(row.result_dir)
        base = {
            "symbol": row.symbol,
            "timeframe": row.timeframe,
            "model": row.model,
            "csv_path": row.csv_path,
            "result_dir": row.result_dir,
        }

        for threshold_bps, ma_window, min_hold_bars, hold_until_opposite, stop_loss_bps, cooldown_bars, long_only in itertools.product(
            thresholds,
            ma_windows,
            min_holds,
            hold_options,
            stop_losses,
            cooldowns,
            long_only_options,
        ):
            metrics = evaluate_backtest(
                csv_path=csv_path,
                result_dir=result_dir,
                target="close",
                seq_len=96,
                pred_step=0,
                target_dim=-1,
                threshold_bps=threshold_bps,
                cost_bps=2.0,
                strategy_mode="stateful",
                ma_window=ma_window,
                min_hold_bars=min_hold_bars,
                hold_until_opposite=hold_until_opposite,
                stop_loss_bps=stop_loss_bps,
                cooldown_bars=cooldown_bars,
                allow_short=not long_only,
                write_outputs=False,
            )
            rows.append(
                {
                    **base,
                    **metrics,
                    "long_only": long_only,
                }
            )

    out_df = pd.DataFrame(rows)
    if args.top_k > 0:
        out_df = (
            out_df.sort_values(["symbol", "timeframe", "model", "cumulative_return"], ascending=[True, True, True, False])
            .groupby(["symbol", "timeframe", "model"], as_index=False, group_keys=False)
            .head(args.top_k)
            .reset_index(drop=True)
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(out_path)


if __name__ == "__main__":
    main()
