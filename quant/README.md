# MT5 + TSLib Workflow

This folder adds a thin quant layer around TSLib instead of changing the training core.

The intended flow is:

1. Export bars from MT5 into a TSLib-friendly CSV.
2. Train one or more forecasting models with `data=custom`.
3. Convert `pred.npy` into trading signals.
4. Compare symbol/timeframe/model combinations by:
   - win rate
   - win/loss ratio
   - max drawdown

## 1. Export MT5 bars

```powershell
python -m quant.export_mt5 `
  --symbol XAUUSD `
  --timeframe H1 `
  --bars 12000 `
  --output-dir dataset/mt5
```

This creates:

- `dataset/mt5/XAUUSD_H1.csv`
- `dataset/mt5/XAUUSD_H1.meta.json`

The CSV layout is compatible with `data=custom`:

- first column: `date`
- last column: target, default `close`
- middle columns: model inputs

If MT5 is not installed yet, use the Yahoo fallback first:

```powershell
python -m quant.export_yfinance `
  --ticker GC=F `
  --symbol XAUUSD `
  --timeframe H1 `
  --period 730d `
  --output-dir dataset/mt5_fallback
```

Useful ticker proxies:

- `EURUSD=X` -> `EURUSD`
- `GBPUSD=X` -> `GBPUSD`
- `GC=F` -> `XAUUSD`
- `NQ=F` -> `NAS100`

## 2. Run a single training job

For trading, start with next-bar forecasting:

```powershell
python run.py `
  --task_name long_term_forecast `
  --is_training 1 `
  --model_id XAUUSD_H1_DLinear `
  --model DLinear `
  --data custom `
  --root_path ./dataset/mt5 `
  --data_path XAUUSD_H1.csv `
  --features MS `
  --target close `
  --freq h `
  --seq_len 128 `
  --label_len 64 `
  --pred_len 1 `
  --enc_in 12 `
  --dec_in 12 `
  --c_out 12 `
  --d_model 64 `
  --d_ff 128 `
  --train_epochs 10 `
  --batch_size 32 `
  --num_workers 0 `
  --inverse
```

Notes:

- `--inverse` matters because the backtest expects real prices, not normalized values.
- `--features MS` is a good default for "multivariate input, close-only decision".
- `pred_len=1` keeps the signal definition clean for a written test.

## 3. Backtest one trained result

```powershell
python -m quant.backtest `
  --csv dataset/mt5/XAUUSD_H1.csv `
  --result-dir results/<setting_name> `
  --target close `
  --seq-len 128 `
  --pred-step 0 `
  --threshold-bps 3 `
  --cost-bps 2
```

Artifacts written into `results/<setting_name>/`:

- `backtest_trades.csv`
- `backtest_summary.json`

## 4. Run a grid search

```powershell
python -m quant.run_grid `
  --data-dir dataset/mt5 `
  --symbols EURUSD,XAUUSD,GBPUSD `
  --timeframes M15,H1,H4 `
  --models DLinear,PatchTST,TimesNet `
  --seq-len 128 `
  --label-len 64 `
  --pred-len 1 `
  --train-epochs 10 `
  --threshold-bps 3 `
  --cost-bps 2 `
  --execute
```

The summary table is saved to `quant/outputs/summary.csv`.

## Suggested exam setup

Use a small but defensible matrix:

- symbols: `EURUSD`, `GBPUSD`, `XAUUSD`, `US100` or `NAS100`
- timeframes: `M15`, `H1`, `H4`
- models: `DLinear`, `PatchTST`, `TimesNet`
- target: next-bar `close`
- signal: long if predicted return > 3 bps, short if < -3 bps, else flat
- cost: 2 bps round-trip

That is enough to answer:

- which symbol is easier to model
- which timeframe is more stable
- whether deep models beat a linear baseline

## Practical recommendation

Start with `DLinear` and `PatchTST`.

- `DLinear` is a strong, cheap baseline.
- `PatchTST` is a more serious deep baseline.
- `TimesNet` is useful as a third comparison point because it is one of the flagship models in this repo.

If compute is limited, test only `XAUUSD` and `EURUSD` on `H1` and `H4` first, then expand.
