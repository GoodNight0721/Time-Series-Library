from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from quant.common import (
    build_setting_name,
    evaluate_backtest,
    get_freq_for_timeframe,
    infer_channel_size,
    normalize_custom_frame,
    sanitize_model_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run symbol/timeframe/model experiments on top of TSLib.")
    parser.add_argument("--data-dir", required=True, help="Directory containing exported CSV files.")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols, for example EURUSD,XAUUSD.")
    parser.add_argument("--timeframes", required=True, help="Comma-separated timeframes, for example M15,H1,H4.")
    parser.add_argument("--models", default="DLinear,PatchTST,TimesNet", help="Comma-separated model list.")
    parser.add_argument(
        "--file-pattern",
        default="{symbol}_{timeframe}.csv",
        help="CSV naming convention inside --data-dir.",
    )
    parser.add_argument("--target", default="close", help="Target column used for forecasting.")
    parser.add_argument("--features", choices=["M", "S", "MS"], default="MS")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--label-len", type=int, default=64)
    parser.add_argument("--pred-len", type=int, default=1, help="Use 1 for direct next-bar trading.")
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--d-layers", type=int, default=1)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--embed", default="timeF")
    parser.add_argument("--des", default="quant")
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold-bps", type=float, default=0.0)
    parser.add_argument("--cost-bps", type=float, default=1.5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu-type", default="cuda")
    parser.add_argument("--use-cpu", action="store_true", help="Force CPU training.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter used to run TSLib.")
    parser.add_argument("--execute", action="store_true", help="Run training/backtest instead of printing commands.")
    parser.add_argument(
        "--summary-out",
        default="quant/outputs/summary.csv",
        help="Summary CSV written after execution.",
    )
    return parser.parse_args()


def build_command(
    *,
    python_exe: str,
    repo_root: Path,
    csv_path: Path,
    model: str,
    model_id: str,
    feature_size: int,
    freq: str,
    args: argparse.Namespace,
) -> list[str]:
    channel_size = 1 if args.features == "S" else feature_size
    cmd = [
        python_exe,
        "run.py",
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--model_id",
        model_id,
        "--model",
        model,
        "--data",
        "custom",
        "--root_path",
        str(csv_path.parent),
        "--data_path",
        csv_path.name,
        "--features",
        args.features,
        "--target",
        args.target,
        "--freq",
        freq,
        "--seq_len",
        str(args.seq_len),
        "--label_len",
        str(args.label_len),
        "--pred_len",
        str(args.pred_len),
        "--enc_in",
        str(channel_size),
        "--dec_in",
        str(channel_size),
        "--c_out",
        str(channel_size),
        "--d_model",
        str(args.d_model),
        "--n_heads",
        str(args.n_heads),
        "--e_layers",
        str(args.e_layers),
        "--d_layers",
        str(args.d_layers),
        "--d_ff",
        str(args.d_ff),
        "--factor",
        str(args.factor),
        "--expand",
        str(args.expand),
        "--d_conv",
        str(args.d_conv),
        "--embed",
        args.embed,
        "--train_epochs",
        str(args.train_epochs),
        "--batch_size",
        str(args.batch_size),
        "--learning_rate",
        str(args.learning_rate),
        "--num_workers",
        str(args.num_workers),
        "--des",
        args.des,
        "--itr",
        str(args.itr),
        "--inverse",
    ]

    if args.use_cpu:
        cmd.append("--no_use_gpu")
    else:
        cmd.extend(["--gpu", str(args.gpu), "--gpu_type", args.gpu_type])

    return cmd


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir)
    summary_rows: list[dict[str, object]] = []

    symbols = [item.strip() for item in args.symbols.split(",") if item.strip()]
    timeframes = [item.strip().upper() for item in args.timeframes.split(",") if item.strip()]
    models = [item.strip() for item in args.models.split(",") if item.strip()]

    for symbol in symbols:
        for timeframe in timeframes:
            csv_name = args.file_pattern.format(symbol=symbol, timeframe=timeframe)
            csv_path = data_dir / csv_name
            if not csv_path.exists():
                summary_rows.append(
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "model": None,
                        "status": f"missing_csv:{csv_path}",
                    }
                )
                continue

            df = normalize_custom_frame(pd.read_csv(csv_path), target=args.target)
            feature_size = infer_channel_size(df, args.features)
            freq = get_freq_for_timeframe(timeframe)

            for model in models:
                model_id = sanitize_model_id(
                    f"{symbol}_{timeframe}_{model}_sl{args.seq_len}_pl{args.pred_len}"
                )
                cmd = build_command(
                    python_exe=args.python,
                    repo_root=repo_root,
                    csv_path=csv_path,
                    model=model,
                    model_id=model_id,
                    feature_size=feature_size,
                    freq=freq,
                    args=args,
                )
                setting = build_setting_name(
                    task_name="long_term_forecast",
                    model_id=model_id,
                    model=model,
                    data="custom",
                    features=args.features,
                    seq_len=args.seq_len,
                    label_len=args.label_len,
                    pred_len=args.pred_len,
                    d_model=args.d_model,
                    n_heads=args.n_heads,
                    e_layers=args.e_layers,
                    d_layers=args.d_layers,
                    d_ff=args.d_ff,
                    expand=args.expand,
                    d_conv=args.d_conv,
                    factor=args.factor,
                    embed=args.embed,
                    distil=True,
                    des=args.des,
                    itr_index=0,
                )
                result_dir = repo_root / "results" / setting

                row: dict[str, object] = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "model": model,
                    "csv_path": str(csv_path),
                    "result_dir": str(result_dir),
                    "command": " ".join(cmd),
                }

                if not args.execute:
                    row["status"] = "planned"
                    summary_rows.append(row)
                    print(row["command"])
                    continue

                try:
                    subprocess.run(cmd, cwd=repo_root, check=True)
                    metrics = evaluate_backtest(
                        csv_path=csv_path,
                        result_dir=result_dir,
                        target=args.target,
                        seq_len=args.seq_len,
                        pred_step=0,
                        target_dim=-1,
                        threshold_bps=args.threshold_bps,
                        cost_bps=args.cost_bps,
                    )
                    row.update(metrics)
                    row["status"] = "ok"
                except Exception as exc:  # noqa: BLE001
                    row["status"] = f"failed:{exc}"

                summary_rows.append(row)

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
