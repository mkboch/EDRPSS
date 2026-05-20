# Robustness and manuscript table scripts

This folder contains additional reproducibility scripts for the OpenClaw/Moltbook corrective-signaling analysis.

These scripts reproduce the added robustness analyses:

1. extended adjusted models and subgroup checks,
2. length-normalized and length-residualized DI models,
3. binary DI robustness,
4. automatic lexical audit of deterministic response labels,
5. manuscript-ready LaTeX table generation.

## Expected local inputs

Run these scripts from the repository/project root after preparing the local data files:

- `Datasets/posts.csv`
- `Datasets/comments.csv`
- `Datasets/agents.csv`
- `results/posts_with_di.csv`
- `Datasets/comments_labeled.csv`

Large raw datasets, derived CSV outputs, generated figures, and temporary analysis folders are intentionally not committed to GitHub.

## Recommended run order

```powershell
python codes\robustness\robustness_extended_checks.py
python codes\robustness\length_normalized_di_models.py
python codes\robustness\binary_di_and_label_audit.py
python codes\robustness\automatic_label_audit.py
python codes\robustness\make_manuscript_tables.py
```

## Main generated outputs

These scripts generate local output folders such as:

- `journal_upgrade_outputs/`
- `journal_upgrade_outputs_round2/`
- `journal_upgrade_outputs_round3/`
- `journal_upgrade_outputs_round4_auto_audit/`
- `journal_upgrade_final_tables/`
- `latex_tables/`

The folder names reflect local analysis stages. They are generated outputs and should not be committed unless a journal or archival release specifically requires them.
