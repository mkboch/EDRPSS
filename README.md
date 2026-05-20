# EDRPSS

EDRPSS is a code-only reproducibility repository for the Moltbook Observatory analysis pipeline. It contains the scripts required to relabel comments, compute Directive Intensity (DI), run coupling and robustness analyses, and regenerate the final figures and result tables used in the paper.

## Repository structure

- `codes/` — Python scripts for labeling, analysis, falsification tests, and figure generation.
- `README.md` — setup and run instructions.
- `requirements.txt` — Python dependencies.
- `.gitignore` — excludes datasets and generated outputs from version control.

## Required scripts

Place the following files in `codes/`:

- `utils_openclaw.py`
- `label_comments_response_type.py`
- `figure_coupling_di_corrective.py`
- `permtest_di_corrective.py`
- `figure_event_aligned_negative_feedback.py`
- `exp_upgrades_nohumans.py`
- `exp_event_aligned_placebo.py`
- `polish_all_figures.py`

## Data

Download the Moltbook Observatory archive from HuggingFace and place the CSV files under `Datasets/`:

- `agents.csv`
- `posts.csv`
- `comments.csv`

Dataset:
[SimulaMet/moltbook-observatory-archive](https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive)

## Install

```bash
pip install -r requirements.txt
```

## Run

Run the pipeline from the repository root:

```bash
python .\codes\label_comments_response_type.py
python .\codes\figure_coupling_di_corrective.py
python .\codes\permtest_di_corrective.py
python .\codes\figure_event_aligned_negative_feedback.py
python .\codes\exp_upgrades_nohumans.py
python .\codes\exp_event_aligned_placebo.py
python .\codes\polish_all_figures.py
```

## Outputs

The scripts generate outputs under:

- `results/`
- `Figures/`

These generated files are intentionally not committed.



@'

## Robustness and manuscript table scripts

Additional robustness scripts are provided under:

- `codes/robustness/`

These scripts reproduce the length-aware and audit analyses used in the manuscript, including length-adjusted DI models, length-normalized DI variants, binary DI robustness, automatic lexical audit of deterministic response labels, and LaTeX table generation.

Recommended run order:

```powershell
python codes\robustness\robustness_extended_checks.py
python codes\robustness\length_normalized_di_models.py
python codes\robustness\binary_di_and_label_audit.py
python codes\robustness\automatic_label_audit.py
python codes\robustness\make_manuscript_tables.py
```

Generated folders such as `journal_upgrade_outputs*`, `journal_upgrade_final_tables/`, and `latex_tables/` are intentionally excluded from version control.
'@ | Add-Content README.md

## Reproducibility policy

This repository is intentionally code-only. Datasets, generated CSV files, text summaries, and rendered figures are excluded from version control so the full analysis can be reproduced locally from the raw archive and scripts.

