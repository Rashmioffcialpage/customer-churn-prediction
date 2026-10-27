# Customer Churn Prediction

A machine learning project exploring customer churn in a fictional
telecommunications dataset. The project compares classification models
and evaluates a selected model on a held-out test set.

## Dataset

IBM's Telco Customer Churn sample describes 7,043 customers of a
fictional telecom company. The target, `Churn`, indicates whether a
customer left.

- [IBM dataset description](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)
- [Source CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

## Workflow

- Inspect data quality and target distribution
- Create a stratified training and test split
- Explore patterns using training data
- Preprocess numeric and categorical features with a pipeline
- Compare logistic regression and random forest using cross-validation
- Select an illustrative decision threshold on training data
- Evaluate the selected model on the held-out test set
- Inspect feature scores and document limitations

## Tools

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Jupyter,
and VS Code.

## Run Locally

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

2. Download the dataset:

   ```bash
   mkdir -p data/raw
   curl -L --fail "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" -o data/raw/telco_churn.csv
   ```

3. Open `notebooks/01_churn_analysis.ipynb` in VS Code, select the
   `.venv` kernel, and choose **Run All**.

The notebook writes charts and evaluation tables to `reports/`.

## Results

Logistic regression achieved a higher mean training cross-validation
F1 score than random forest (0.592 vs. 0.564), using their default
classification thresholds.

After selecting a threshold using training data, the selected model
achieved the following on the held-out test set:

| Metric | Test score |
| --- | ---: |
| Accuracy | 0.765 |
| Precision | 0.544 |
| Recall | 0.706 |
| F1 | 0.615 |
| ROC AUC | 0.842 |
| Average precision | 0.633 |

At the selected threshold, the model identified approximately 71% of
test-set churners. Approximately 54% of customers it predicted would
churn actually did so.

These results describe a held-out sample from a fictional dataset.

## Limitations

- The dataset is fictional and represents a single sample.
- The random test split does not establish future performance.
- Feature importance and coefficients do not establish causality.
- The threshold is illustrative; a real deployment would require
  business cost, privacy, fairness, and monitoring considerations.

## Author

Rashmi Thimmaraju · [GitHub](https://github.com/Rashmioffcialpage)