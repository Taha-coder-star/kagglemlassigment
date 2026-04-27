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

- `lgbm_holdout`
- `lgbm_scipy`
- `blend_holdout`

Each submission is saved separately in `outputs/submissions/`.

## 5. Download a submission

```python
from google.colab import files
files.download("outputs/submissions/lgbm_holdout.csv")
```

Start by submitting `lgbm_holdout.csv`. Then try `lgbm_scipy.csv` and
`blend_holdout.csv` if you still have Kaggle submissions available.

## Optional Commands

Run only one experiment:

```python
!python run_colab_pipeline.py --experiments lgbm_holdout
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
!python run_colab_pipeline.py --experiments lgbm_holdout,lgbm_high_15
```
