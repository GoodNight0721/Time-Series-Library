from __future__ import annotations

import argparse
from pathlib import Path

from quant.common import evaluate_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert TSLib forecast outputs into simple trading metrics.")
    parser.add_argument("--csv", required=True, help="Path to the custom CSV used for training.")
    parser.add_argument("--result-dir", required=True, help="Directory containing pred.npy and true.npy.")
    parser.add_argument("--target", default="close", help="Target column name.")
    parser.add_argument("--seq-len", type=int, required=True, help="Must match the training seq_len.")
    parser.add_argument("--pred-step", type=int, default=0, help="Use 0 for next-bar trading.")
    parser.add_argument(
        "--target-dim",
        type=int,
        default=-1,
        help="Channel index inside pred.npy. Keep -1 when target is the last column.",
    )
    parser.add_argument("--threshold-bps", type=float, default=0.0, help="Signal threshold in basis points.")
    parser.add_argument("--cost-bps", type=float, default=1.5, help="Round-trip trading cost in basis points.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_backtest(
        csv_path=Path(args.csv),
        result_dir=Path(args.result_dir),
        target=args.target,
        seq_len=args.seq_len,
        pred_step=args.pred_step,
        target_dim=args.target_dim,
        threshold_bps=args.threshold_bps,
        cost_bps=args.cost_bps,
    )

    for key in (
        "trade_count",
        "coverage",
        "win_rate",
        "win_loss_ratio",
        "profit_factor",
        "cumulative_return",
        "max_drawdown",
        "pred_direction_acc",
    ):
        print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
