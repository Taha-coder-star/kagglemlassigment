# Colab Run Pipeline

Use this after pushing the repo to GitHub.

## 1. Clone the repo

```python
!git clone https://github.com/YOUR_USERNAME/irrigation-kaggle-assignment.git
%cd irrigation-kaggle-assignment
```

If already cloned:

```python
%cd irrigation-kaggle-assignment
!git pull
```

## 2. Install dependencies

```python
%pip install -q -r requirements.txt
```

## 3. Add data

If the CSV files are in Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p data
!cp /content/drive/MyDrive/irrigation-kaggle-assignment/data/*.csv data/
!ls -lh data
```

The `data` folder must contain:

```text
train.csv
test.csv
sample_submission.csv
```

## 4. Run the pipeline

```python
!python run_colab_pipeline.py
```

This runs:

- `legacy_best`

The submission is saved in `outputs/submissions/legacy_best.csv`.

## 5. Download a submission

```python
from google.colab import files
files.download("outputs/submissions/legacy_best.csv")
```

Start by submitting `legacy_best.csv`. This is the safer fallback because the
newer LightGBM/blend experiments scored worse than the earlier improved model
on the public leaderboard.

## Optional Commands

Run only one experiment:

```python
!python run_colab_pipeline.py --experiments legacy_best
```

Run only the smoke test:

```python
!python run_colab_pipeline.py --smoke-only
```

Skip the smoke test:

```python
!python run_colab_pipeline.py --skip-smoke-test
```

Run a specific set:

```python
!python run_colab_pipeline.py --experiments legacy_best,lgbm_holdout
```
