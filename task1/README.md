# Task 1

Use this directory for topology reconstruction experiments, notes, and task-specific scripts.

`q1_1a_summary.py` implements Question 1.1(a): it computes per-router flow counts, flow-duration summary statistics, top-5 destination ports, and forward/backward packet ratios, then writes:

- `outputs/task1/tables/q1_1a_router_summary.csv`
- `outputs/task1/tables/q1_1a_discussion.md`
- `outputs/task1/figures/q1_1a_router_summary_dashboard.png`
- `outputs/task1/tables/q1_1b_volume_summary.csv`
- `outputs/task1/tables/q1_1b_discussion.md`
- `outputs/task1/figures/q1_1b_volume_distributions.png`
- `outputs/task1/tables/q1_1c_feature_variance_ranking.csv`
- `outputs/task1/tables/q1_1c_top10_features.csv`
- `outputs/task1/tables/q1_1c_discussion.md`
- `outputs/task1/figures/q1_1c_top10_feature_variance.png`
- `outputs/task1/tables/q1_2a_router_feature_profiles.csv`
- `outputs/task1/tables/q1_2a_router_similarity_matrix.csv`
- `outputs/task1/tables/q1_2a_methodology.md`
- `outputs/task1/figures/q1_2a_router_similarity_heatmap.png`
- `outputs/task1/tables/q1_2b_knn_edges.csv`
- `outputs/task1/tables/q1_2b_mst_edges.csv`
- `outputs/task1/tables/q1_2b_graph_summary.csv`
- `outputs/task1/tables/q1_2b_discussion.md`
- `outputs/task1/figures/q1_2b_adjacency_graphs.png`
- `outputs/task1/tables/q1_2c_ip_jaccard_matrix.csv`
- `outputs/task1/tables/q1_2c_tuple_jaccard_matrix.csv`
- `outputs/task1/tables/q1_2c_pairwise_overlap_summary.csv`
- `outputs/task1/tables/q1_2c_refined_graph_edges.csv`
- `outputs/task1/tables/q1_2c_discussion.md`
- `outputs/task1/figures/q1_2c_ip_jaccard_heatmap.png`
- `outputs/task1/figures/q1_2c_tuple_jaccard_heatmap.png`
- `outputs/task1/figures/q1_2c_refined_mst.png`
- `outputs/task1/tables/q1_3a_pair_scores.csv`
- `outputs/task1/tables/q1_3a_inferred_topology_edges.csv`
- `outputs/task1/tables/q1_3a_discussion.md`
- `outputs/task1/figures/q1_3a_inferred_topology.png`
- `outputs/task1/tables/q1_3b_router_metrics.csv`
- `outputs/task1/tables/q1_3b_graph_summary.csv`
- `outputs/task1/tables/q1_3b_removal_impact.csv`
- `outputs/task1/tables/q1_3b_discussion.md`
- `outputs/task1/figures/q1_3b_graph_properties.png`
- `outputs/task1/tables/q1_3c_comparison_summary.csv`
- `outputs/task1/tables/q1_3c_structural_feature_comparison.csv`
- `outputs/task1/tables/q1_3c_discussion.md`
- `outputs/task1/figures/q1_3c_comparison_dashboard.png`

Run it from the project root with:

```bash
python3.11 task1/q1_1a_summary.py
python3.11 task1/q1_1a_plots.py
python3.11 task1/q1_1b_volume.py
python3.11 task1/q1_1c_feature_variance.py
python3.11 task1/q1_2a_similarity.py
python3.11 task1/q1_2b_graphs.py
python3.11 task1/q1_2c_ip_overlap.py
python3.11 task1/q1_3a_topology.py
python3.11 task1/q1_3b_graph_properties.py
python3.11 task1/q1_3c_geant_comparison.py
```
