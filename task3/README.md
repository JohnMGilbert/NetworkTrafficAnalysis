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

If you omit the explicit paths, the default workflow now:

- trains on the labeled corpus under `data/test/`
- evaluates the fitted models on the full labeled corpus under `data/raw_labeled/`

If you instead have a pre-split pair of labeled files in `data/processed/` whose filenames already contain `train` and `test`, you can still point the script at them explicitly or rely on auto-discovery:

```bash
/opt/homebrew/bin/python3.11 task3/q3_1a_baselines.py
```

If neither the dedicated corpora nor a pre-split pair are available, the script still falls back to recursively loading `data/raw_labeled/` and creating a reproducible stratified train/test split automatically.

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

By default, the script tries to reuse the strongest tree-based baseline from Question 3.1(a). If the serialized baseline model is missing, it retrains the selected tree model on `data/test/` and then extracts feature importances.

Like Question 3.1(a), this script defaults to the `data/test/` training corpus plus the full `data/raw_labeled/` evaluation corpus, with the older auto-split path still available as a fallback.

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

## Question 3.2(a)

Run the imbalance-handling comparison with:

```bash
/opt/homebrew/bin/python3.11 task3/q3_2a_imbalance_strategies.py
```

The script retrains the best baseline family for this project, `RandomForest`, under four conditions:

- no balancing
- selective `SMOTE` oversampling
- `class_weight="balanced"`
- hybrid `SMOTE + random undersampling`

By default, each strategy is fit on `data/test/` and evaluated on the full labeled corpus in `data/raw_labeled/`.

Because the assignment warns that naive SMOTE over the full 6.8M-flow corpus is computationally prohibitive, the SMOTE-based strategies only oversample classes whose training counts fall below `--minority-target-count` (default `25000`). The hybrid strategy then caps the largest classes with random undersampling via `--hybrid-majority-cap` (default `250000`).

Useful optional flags:

- `--minority-target-count 25000` to control which classes are treated as minority classes for SMOTE-based strategies
- `--hybrid-majority-cap 250000` to control how aggressively the hybrid strategy undersamples the largest classes
- `--max-train-rows 50000 --max-test-rows 10000` for faster smoke tests while developing

Artifacts written by default:

- `outputs/task3/tables/q3_2a_imbalance_summary.csv`
- `outputs/task3/tables/q3_2a_per_class_metrics.csv`
- `outputs/task3/tables/q3_2a_sampling_plan.csv`
- `outputs/task3/tables/q3_2a_strategy_sampling.csv`
- `outputs/task3/tables/q3_2a_summary.json`
- `outputs/task3/tables/q3_2a_report.md`
- `outputs/models/task3/q3_2a_randomforest_baseline.joblib`
- `outputs/models/task3/q3_2a_randomforest_smote.joblib`
- `outputs/models/task3/q3_2a_randomforest_class_weight.joblib`
- `outputs/models/task3/q3_2a_randomforest_hybrid.joblib`

## Question 3.2(b)

Run the rare-class analysis workflow with:

```bash
/opt/homebrew/bin/python3.11 task3/q3_2b_rare_class_analysis.py
```

This script reads the 3.2(a) outputs, creates the grouped bar chart requested by the assignment for the three rare classes:

- `Web-command-injection`
- `Web-sql-injection`
- `Infiltration-mitm`

It also summarizes whether the rare-class improvements come with any measurable cost on the non-rare classes.

Artifacts written by default:

- `outputs/task3/figures/q3_2b_rare_class_f1.png`
- `outputs/task3/tables/q3_2b_rare_class_f1.csv`
- `outputs/task3/tables/q3_2b_rare_class_summary.csv`
- `outputs/task3/tables/q3_2b_majority_tradeoff.csv`
- `outputs/task3/tables/q3_2b_majority_deltas.csv`
- `outputs/task3/tables/q3_2b_summary.json`
- `outputs/task3/tables/q3_2b_report.md`

## Question 3.3

Run the advanced-model workflow with:

```bash
/opt/homebrew/bin/python3.11 task3/q3_3_advanced_model.py
```

The script builds a tuned hybrid model designed to fix the main failure mode we observed in Questions 3.1 and 3.2:

- a class-weighted `RandomForest` backbone for the full 11-class decision
- a binary web-family detector that flags likely web traffic
- a scaled `kNN` specialist that separates `Web-xss`, `Web-sql-injection`, and `Web-command-injection`
- four engineered ratio / burst features carried over from Task 2

The workflow automatically:

- creates an internal train/validation split for hyperparameter selection
- searches the backbone, web-detector, and web-specialist hyperparameters
- retrains the best hybrid model on the full Task 3 training set
- reports all Question 3.1(a) metrics on the full evaluation corpus
- compares the result with the best Question 3.2 strategy when those outputs are available
- runs a 3-component ablation study
- writes a report plus an architecture diagram

Useful optional flags:

- `--tuning-max-train-rows 600000` to control how many rows are used for hyperparameter selection
- `--max-train-rows 120000 --max-test-rows 25000` for a quicker smoke test while developing
- `--validation-size 0.15` to change the held-out fraction used for tuning

Artifacts written by default:

- `outputs/task3/figures/q3_3_advanced_architecture.svg`
- `outputs/task3/tables/q3_3_advanced_summary.csv`
- `outputs/task3/tables/q3_3_per_class_metrics.csv`
- `outputs/task3/tables/q3_3_hyperparameter_search.csv`
- `outputs/task3/tables/q3_3_ablation_results.csv`
- `outputs/task3/tables/q3_3_comparison_vs_q3_2.csv` when the Question 3.2 outputs are present and the evaluation set is unchanged
- `outputs/task3/tables/q3_3_per_class_comparison_vs_q3_2.csv` under the same comparison conditions
- `outputs/task3/tables/q3_3_summary.json`
- `outputs/task3/tables/q3_3_report.md`
- `outputs/models/task3/q3_3_hybrid_advanced_model.joblib`

## Question 3.4(a)

Run the router-level evaluation with:

```bash
/opt/homebrew/bin/python3.11 task3/q3_4a_router_level_analysis.py
```

By default, the script:

- evaluates the selected model on the full labeled evaluation corpus under `data/raw_labeled/`
- auto-selects the best available model artifact across Questions 3.1(a), 3.2(a), and 3.3 using macro F1
- reuses the saved hybrid holdout manifest only when older Question 3.1(a) outputs require it
- evaluates that model router-by-router on the chosen evaluation set
- reports per-router summary metrics and per-router/per-class F1-scores
- relates the hardest/easiest routers to their evaluation-set class distributions

Artifacts written by default:

- `outputs/task3/figures/q3_4a_router_class_f1_heatmap.png`
- `outputs/task3/figures/q3_4a_router_class_distribution.png`
- `outputs/task3/tables/q3_4a_router_summary.csv`
- `outputs/task3/tables/q3_4a_router_class_metrics.csv`
- `outputs/task3/tables/q3_4a_router_class_f1_matrix.csv`
- `outputs/task3/tables/q3_4a_router_class_distribution.csv`
- `outputs/task3/tables/q3_4a_router_mix_summary.csv`
- `outputs/task3/tables/q3_4a_hardest_router_class_pairs.csv`
- `outputs/task3/tables/q3_4a_summary.json`
- `outputs/task3/tables/q3_4a_report.md`

## Question 3.4(b)

Run the cross-router generalization experiment with:

```bash
/opt/homebrew/bin/python3.11 task3/q3_4b_cross_router_generalization.py
```

By default, the script:

- auto-selects the best available model artifact across Questions 3.1(a), 3.2(a), and 3.3
- retrains that model family on routers `D1` through `D7`
- evaluates on the unseen routers `D8` through `D10`
- compares the result against the standard Task 3 evaluation metrics for the same model family
- highlights classes that appear only in the unseen routers and reports their impact on generalization

Useful optional flags:

- `--train-routers D1,D2,D3,D4,D5,D6,D7`
- `--test-routers D8,D9,D10`
- `--max-train-rows 200000 --max-test-rows 50000` for quicker smoke tests

Artifacts written by default:

- `outputs/task3/figures/q3_4b_overall_metric_comparison.png`
- `outputs/task3/tables/q3_4b_cross_router_summary.csv`
- `outputs/task3/tables/q3_4b_cross_router_per_class_metrics.csv`
- `outputs/task3/tables/q3_4b_comparison_vs_standard.csv`
- `outputs/task3/tables/q3_4b_per_class_comparison_vs_standard.csv`
- `outputs/task3/tables/q3_4b_class_coverage.csv`
- `outputs/task3/tables/q3_4b_router_summary.csv`
- `outputs/task3/tables/q3_4b_router_class_metrics.csv`
- `outputs/task3/tables/q3_4b_router_mix_summary.csv`
- `outputs/task3/tables/q3_4b_summary.json`
- `outputs/task3/tables/q3_4b_report.md`
- `outputs/models/task3/q3_4b_cross_router_model.joblib`
