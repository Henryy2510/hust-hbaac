# HUST Demand Forecasting Baseline

56-day SKU demand forecasting pipeline for:
- `train.csv` (history)
- `sample_submission.csv` (required submission shape)

## Setup

```bash
python -m pip install -U pip
python -m pip install -e .
```

## Run Training + Submission

```bash
python train_model.py \
  --train train.csv \
  --sample sample_submission.csv \
  --config config.yaml \
  --out submission.csv \
  --metrics-out metrics.json \
  --importance-out feature_importance.csv
```

## Run Local Backtest

```bash
python validate.py \
  --train train.csv \
  --config config.yaml \
  --cutoffs 2025-07-01,2025-08-01 \
  --out backtest_metrics.json
```

## Notes

- Quantity is aggregated to daily net demand per SKU.
- Negative net demand can occur in history due to returns; predictions are clipped to non-negative.
- Submission is generated from `sample_submission.csv` ids and preserves exact row set.
