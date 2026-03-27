# 提交说明

这个目录是本次笔试的第二版提交。

和第一版相比，这一版把策略从“单根 K 线开平仓”改成了状态化持仓，并补充了：

- 更高阈值开仓
- 持有到反向信号
- 止损
- 冷却期
- 均线过滤测试

建议先看：

1. `answer/tslib_mt5_answer.pdf`

里面是完整答卷，包含策略修改思路、改进前后对照和最终结果。

目录说明：

- `answer/`
  - 笔试答卷 PDF
- `code/quant/`
  - 数据导出、训练、回测和策略参数搜索脚本
- `results/`
  - 本次答卷引用的核心结果表

关键代码：

- `export_mt5.py`
- `run_grid.py`
- `backtest.py`
- `common.py`
- `strategy_sweep.py`
- `mql5/ExportRatesCsv.mq5`

核心结果表：

- `strategy_before_after_selected.csv`
- `strategy_submission_selected.csv`
- `strategy_submission_best.csv`
- `strategy_submission_ma_examples.csv`
- `strategy_sweep_best_credible.csv`

完整仓库地址：

`https://github.com/GoodNight0721/Time-Series-Library`

说明：

- 原始 MT5 行情 CSV 没有放进提交包。
- 如果需要完整复核，可以按仓库脚本从本地 MT5 终端重新导出数据，再复现训练与回测。
