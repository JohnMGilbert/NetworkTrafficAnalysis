# Task 1

Use this directory for topology reconstruction experiments, notes, and task-specific scripts.

`q1_1a_summary.py` implements Question 1.1(a): it computes per-router flow counts, flow-duration summary statistics, top-5 destination ports, and forward/backward packet ratios, then writes:

- `outputs/task1/tables/q1_1a_router_summary.csv`
- `outputs/task1/tables/q1_1a_discussion.md`
- `outputs/task1/figures/q1_1a_router_summary_dashboard.png`
- `outputs/task1/tables/q1_1b_volume_summary.csv`
- `outputs/task1/tables/q1_1b_discussion.md`
- `outputs/task1/figures/q1_1b_volume_distributions.png`

Run it from the project root with:

```bash
python3.11 task1/q1_1a_summary.py
python3.11 task1/q1_1a_plots.py
python3.11 task1/q1_1b_volume.py
```
