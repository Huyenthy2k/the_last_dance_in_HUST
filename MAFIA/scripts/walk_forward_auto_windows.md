Walk-forward auto windows (defaults after update)

- `--walkforward-num-windows <= 0` ⇒ auto-compute maximum windows until `--walkforward-end-date` (inclusive). Defaults: start=2017-01-01, end=2023-12-31, step=1y, train/valid/test=3/1/1y.
- Chain checkpoint_best_valid between windows (buffer reset on resume).
- Example (auto windows to 2023-12-31):
  ```
  python scripts/auto_pipeline.py --num-seeds 10 --base-seed 2025
  ```
  (num-windows=0 by default → auto windows; hparam search still 10 trials x 5 mini-epochs; walkforward-epochs=50).
