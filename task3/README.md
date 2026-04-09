# Task 3

Use this directory for supervised model training, imbalance handling experiments, and router-level evaluation.

## Question 3.1(a)

Install the Task 3 dependencies into a Python 3.11 environment before running the script:

```bash
/opt/homebrew/bin/python3.11 -m pip install -r requirements.txt
```

Run the baseline supervised-model workflow with:

```bash
/opt/homebrew/bin/python3.11 task3/q3_1a_baselines.py --train-path data/processed/<labeled-train-file> --test-path data/processed/<labeled-test-file>
```

If your labeled files in `data/processed/` already contain `train` and `test` in their filenames, the explicit paths are optional:

```bash
/opt/homebrew/bin/python3.11 task3/q3_1a_baselines.py
```

If no pre-split files exist yet, the script also falls back to recursively loading the labeled CSVs under `data/raw_labeled/` and creating a reproducible train/test split automatically. The default is now a stricter `source_file` holdout, which keeps whole labeled files together instead of splitting rows from the same source across train and test.

The script expects a label column such as `label`, `class`, `target`, `attack_label`, or `attack_type`. It trains:

- Random Forest (`>=100` trees; default `200`)
- LightGBM when available, otherwise XGBoost
- Multi-Layer Perceptron with two hidden layers

If LightGBM and XGBoost are both unavailable because native OpenMP libraries are missing on macOS, the script falls back to scikit-learn `HistGradientBoostingClassifier` so the workflow can still complete.

Artifacts written by default:

- `outputs/task3/tables/q3_1a_baseline_summary.csv`
- `outputs/task3/tables/q3_1a_per_class_metrics.csv`
- `outputs/task3/tables/q3_1a_summary.json`
- `outputs/task3/tables/q3_1a_report.md`
- `outputs/models/task3/q3_1a_randomforest_baseline.joblib`
- `outputs/models/task3/q3_1a_lightgbm_baseline.joblib` or `outputs/models/task3/q3_1a_xgboost_baseline.joblib`
- `outputs/models/task3/q3_1a_mlp_baseline.joblib`

The generated summary table reports all metrics required by Question 3.1(a):

- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)
- F1-score (weighted)
- Training time

## Question 3.1(b)

Run the feature-importance workflow with:

```bash
/opt/homebrew/bin/python3.11 task3/q3_1b_feature_importance.py --train-path data/processed/<labeled-train-file> --test-path data/processed/<labeled-test-file>
```

By default, the script tries to reuse the strongest tree-based baseline from Question 3.1(a). If the serialized baseline model is missing, it retrains the selected tree model with the same baseline configuration and then extracts feature importances.

Like Question 3.1(a), this script can also auto-build a train/test split from `data/raw_labeled/` when explicit or pre-split labeled datasets are not available, and it uses the same `source_file` holdout by default.

Artifacts written by default:

- `outputs/task3/figures/q3_1b_top20_feature_importance.png`
- `outputs/task3/tables/q3_1b_top20_feature_importance.csv`
- `outputs/task3/tables/q3_1b_feature_pair_separation.csv`
- `outputs/task3/tables/q3_1b_summary.json`
- `outputs/task3/tables/q3_1b_report.md`

The generated report discusses:

- the top-20 most important features for the chosen tree model
- whether those features align with the Task 2 engineered-feature ideas
- which class pairs each important feature separates most strongly in the labeled training set

## Question 3.1(c)

Run the confusion-matrix workflow with:

```bash
/opt/homebrew/bin/python3.11 task3/q3_1c_confusion_matrix.py
```

By default, the script selects the best baseline model from the Question 3.1(a) summary, loads the serialized pipeline when available, and evaluates it on the same deterministic test split.

Artifacts written by default:

- `outputs/task3/figures/q3_1c_confusion_matrix.png`
- `outputs/task3/tables/q3_1c_confusion_matrix.csv`
- `outputs/task3/tables/q3_1c_confusion_matrix_normalized.csv`
- `outputs/task3/tables/q3_1c_top_confusions.csv`
- `outputs/task3/tables/q3_1c_summary.json`
- `outputs/task3/tables/q3_1c_report.md`

The generated report highlights the most commonly confused class pairs and provides short networking-oriented hypotheses for why those confusions occur.
