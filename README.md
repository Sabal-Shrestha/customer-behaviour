# Analysing customer behaviour using transaction data

This project adds a simple and reproducible baseline to predict whether a customer will make a purchase within the next 30 days, using the existing dataset in `data/transaction-dataset.csv`.

## What’s included

- Notebook: `notebooks/next_purchase_30d.ipynb`
	- Loads and profiles the dataset
	- Time-aware train/val/test splits with a safe 30-day label horizon
	- One snapshot per customer per split (the customer's last transaction within that split window)
	- Labels: next purchase within 30 days
	- Features: simple RFM-style stats over the prior 90 days + category/payment counts
	- Model: Logistic Regression baseline
	- Metrics: ROC-AUC, PR-AUC, Recall@Top10%; saved to `outputs/`

## Setup

1) Create a virtual environment (optional, recommended)

```zsh
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies

```zsh
pip install -r requirements.txt
```

3) Open the notebook in VS Code

```zsh
code notebooks/next_purchase_30d.ipynb
```

Run the cells from top to bottom. The notebook will:
- Create `outputs/` and `logs/` if missing
- Install/verify Python packages in the active kernel
- Save metrics and plots to `outputs/`

## Data

Place the CSV at `data/transaction-dataset.csv` (already present). The notebook automatically detects the date range and computes safe snapshot windows so labels don’t peek beyond the dataset end (we only take snapshot dates that are at least 30 days before the dataset's last date).

## Plain-English glossary

- snapshot_date: the reference date for a customer (their most recent transaction inside a split window). This was previously called anchor_date.
- label_next_30d: 1 if the customer purchases within 30 days after the snapshot_date, else 0.
- predicted_probability_30d: the model’s estimated probability that a customer will purchase within 30 days.
- is_top_10_percent: whether the customer is in the top 10% by predicted_probability_30d in a given split.

## What gets saved where (outputs map)

- metrics_next_purchase_30d.json: created in Section 8 (baseline metrics) of the notebook.
- metrics_next_purchase_30d_refined.json: created in Section 17 (refined features retrain).
- curves_val.png and curves_test.png: created in Sections 9/16 (ROC and PR plots for Validation/Test).
- snapshots_train.csv, snapshots_val.csv, snapshots_test.csv: created in Section 10 (one snapshot per customer per split).
- predictions_val.csv, predictions_test.csv: created in Section 18 (ranked customers with predicted_probability_30d and is_top_10_percent).

## Notes & next steps

- The first version uses one snapshot per customer per split to keep training size moderate.
 - You can expand to multiple snapshots per customer (e.g., treat each transaction date as a snapshot) once you’re happy with the baseline.
- Next incremental features: more time-windowed aggregates, store/location signals, and a gradient-boosting model.
