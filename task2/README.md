# Task 2

Use this directory for preprocessing, feature engineering, clustering, and attack discovery analysis.

Suggested contents:

- sampling and preprocessing workflows
- clustering experiment logs
- cluster interpretation notes and tables

## Question 2.1(a)

Run the preprocessing pipeline with:

```bash
python3 task2/q2_1a_preprocessing.py
```

Artifacts written by default:

- `data/processed/task2_preprocessed_standardized.parquet`
- `outputs/task2/tables/q2_1a_feature_audit.csv`
- `outputs/task2/tables/q2_1a_before_stats.csv`
- `outputs/task2/tables/q2_1a_after_stats.csv`
- `outputs/task2/tables/q2_1a_scaler_parameters.csv`
- `outputs/task2/tables/q2_1a_dataset_summary.json`
- `outputs/task2/tables/q2_1a_preprocessing_report.md`

## Question 2.1(b)

Run the dimensionality reduction workflow with:

```bash
/opt/homebrew/bin/python3.11 task2/q2_1b_dimensionality_reduction.py
```

Artifacts written by default:

- `outputs/task2/figures/q2_1b_pca_cumulative_variance.png`
- `outputs/task2/figures/q2_1b_umap_router_embedding.png`
- `outputs/task2/tables/q2_1b_pca_explained_variance.csv`
- `outputs/task2/tables/q2_1b_umap_sample_allocation.csv`
- `outputs/task2/tables/q2_1b_umap_embedding.csv`
- `outputs/task2/tables/q2_1b_summary.json`
- `outputs/task2/tables/q2_1b_report.md`
- `data/interim/task2_q2_1b_umap_sample.parquet`

## Question 2.1(c)

Run the feature-engineering workflow with:

```bash
/opt/homebrew/bin/python3.11 task2/q2_1c_feature_engineering.py
```

Artifacts written by default:

- `outputs/task2/figures/q2_1c_engineered_feature_distributions.png`
- `outputs/task2/tables/q2_1c_engineered_feature_definitions.csv`
- `outputs/task2/tables/q2_1c_engineered_feature_summary.csv`
- `outputs/task2/tables/q2_1c_sample_allocation.csv`
- `outputs/task2/tables/q2_1c_preview_cluster_feature_means.csv`
- `outputs/task2/tables/q2_1c_summary.json`
- `outputs/task2/tables/q2_1c_report.md`
- `data/interim/task2_q2_1c_engineered_sample.parquet`

## Question 2.2(a)

Run the clustering workflow with:

```bash
/opt/homebrew/bin/python3.11 task2/q2_2a_clustering.py
```

Artifacts written by default:

- `outputs/task2/figures/q2_2a_minibatchkmeans_clusters.png`
- `outputs/task2/figures/q2_2a_hdbscan_clusters.png`
- `outputs/task2/figures/q2_2a_gaussianmixture_clusters.png`
- `outputs/task2/tables/q2_2a_cluster_assignments.csv`
- `outputs/task2/tables/q2_2a_cluster_sizes.csv`
- `outputs/task2/tables/q2_2a_clustering_summary.csv`
- `outputs/task2/tables/q2_2a_hyperparameter_search.csv`
- `outputs/task2/tables/q2_2a_summary.json`
- `outputs/task2/tables/q2_2a_report.md`

## Question 2.2(b)

Run the internal-validation workflow with:

```bash
/opt/homebrew/bin/python3.11 task2/q2_2b_evaluation.py
```

Artifacts written by default:

- `outputs/task2/tables/q2_2b_internal_validation.csv`
- `outputs/task2/tables/q2_2b_summary.json`
- `outputs/task2/tables/q2_2b_report.md`

## Question 2.2(c)

Run the HDBSCAN cluster-interpretation workflow with:

```bash
/opt/homebrew/bin/python3.11 task2/q2_2c_hdbscan_analysis.py
```

Artifacts written by default:

- `outputs/task2/tables/q2_2c_hdbscan_cluster_statistics.csv`
- `outputs/task2/tables/q2_2c_hdbscan_cluster_mapping.csv`
- `outputs/task2/tables/q2_2c_hdbscan_confusion_style_mapping.csv`
- `outputs/task2/tables/q2_2c_hdbscan_summary.json`
- `outputs/task2/tables/q2_2c_hdbscan_report.md`

## Question 2.3(a)

Run the discoverability-analysis workflow with:

```bash
/opt/homebrew/bin/python3.11 task2/q2_3a_discoverability.py
```

Artifacts written by default:

- `outputs/task2/tables/q2_3a_discoverability_ranking.csv`
- `outputs/task2/tables/q2_3a_summary.json`
- `outputs/task2/tables/q2_3a_report.md`
