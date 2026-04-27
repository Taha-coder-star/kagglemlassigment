# Irrigation Kaggle Assignment

Kaggle Playground Series S6E4 pipeline for predicting `Irrigation_Need`
(`Low`, `Medium`, `High`) with macro F1.

## Colab Quick Start

Clone the repo in Colab, install dependencies, copy the Kaggle CSVs into
`data/`, then run:

```python
%pip install -q -r requirements.txt
!python run_colab_pipeline.py
```

The runner creates separate submission files in:

```text
outputs/submissions/
```

Default experiment:

- `legacy_best.csv`

Start by submitting `legacy_best.csv` to Kaggle. This recreates the older
HistGBM setup that beat the newer LightGBM/blend experiments on the public
leaderboard.

See [COLAB_RUN.md](COLAB_RUN.md) for the full Colab setup.

## Useful Commands

Smoke test only:

```bash
python run_colab_pipeline.py --smoke-only
```

Run one experiment:

```bash
python run_colab_pipeline.py --experiments legacy_best
```

Run the core model directly:

```bash
python src/improved_pipeline.py --holdout-only --model lgbm
```
