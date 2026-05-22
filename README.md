# EDRPSS

EDRPSS is a code-only reproducibility repository for the Moltbook Observatory analysis pipeline. It contains the scripts required to relabel comments, compute Directive Intensity (DI), run raw association and robustness analyses, perform human-validation checks, and regenerate the figures and result tables used in the manuscript.

## Repository structure

- `codes/` — Python scripts for labeling, analysis, falsification tests, figure generation, and human-validation checks.
- `codes/robustness/` — additional robustness and manuscript-table scripts, including length-aware DI analyses, automatic lexical audit, and LaTeX table generation.
- `README.md` — setup and run instructions.
- `requirements.txt` — Python dependencies.
- `.gitignore` — excludes datasets, generated outputs, annotation files, figures, and temporary artifacts from version control.

## Reproducibility policy

This repository is intentionally code-only. Raw datasets, derived CSV files, annotation files, generated result tables, rendered figures, ZIP archives, and temporary analysis folders are excluded from version control. The full analysis can be reproduced locally from the raw Moltbook Observatory Archive and the scripts in this repository.

## Data

Download the Moltbook Observatory Archive from Hugging Face and place the required CSV files under `Datasets/`:

- `Datasets/agents.csv`
- `Datasets/posts.csv`
- `Datasets/comments.csv`

Dataset:

- [SimulaMet/moltbook-observatory-archive](https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive)

Some later scripts also expect derived local artifacts produced by earlier steps:

- `Datasets/comments_labeled.csv`
- `results/posts_with_di.csv`

These derived files are generated locally and are not committed.

## Install

Install Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```

## Core analysis scripts

The core pipeline scripts are kept under `codes/`. Depending on the exact local file organization, the main analysis uses scripts such as:

- `codes/utils_openclaw.py`
- `codes/label_comments_response_type.py`
- `codes/figure_coupling_di_corrective.py`
- `codes/permtest_di_corrective.py`
- `codes/figure_event_aligned_negative_feedback.py`
- `codes/exp_upgrades_nohumans.py`
- `codes/exp_event_aligned_placebo.py`
- `codes/polish_all_figures.py`

Recommended run order from the repository root:

```powershell
python codes\label_comments_response_type.py
python codes\figure_coupling_di_corrective.py
python codes\permtest_di_corrective.py
python codes\figure_event_aligned_negative_feedback.py
python codes\exp_upgrades_nohumans.py
python codes\exp_event_aligned_placebo.py
python codes\polish_all_figures.py
```

## Robustness and manuscript-table scripts

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


The folders are generated locally and intentionally excluded from version control.

## Human-validation scripts

The manuscript includes a small two-annotator validation study for deterministic response labels. The code for that workflow is included under `codes/`.

Human-validation scripts include:

- `codes/make_human_validation_sample.py`
- `codes/check_human_validation_annotations.py`
- `codes/compute_human_validation_metrics.py`

The typical workflow is:

```powershell
python codes\make_human_validation_sample.py
python codes\check_human_validation_annotations.py
python codes\compute_human_validation_metrics.py
```

The human-validation scripts expect local annotation files and hidden key files generated from `comments_labeled.csv`. The annotation CSV files, hidden machine-label key, validation results, and generated LaTeX tables are not committed to GitHub.

Generated local files/folders may include:

- `human_validation_sample_400_BLINDED.csv`
- `human_validation_sample_400_KEY_DO_NOT_SHARE.csv`
- `human_validation_sample_400_ANNOTATED_annotator1.csv`
- `human_validation_sample_400_ANNOTATED_annotator2.csv`
- `human_validation_checks/`
- `human_validation_results/`
- `table_human_validation.tex`

These files are excluded from version control because they are data or generated artifacts.

## Outputs

The scripts generate analysis outputs locally under folders such as:

- `results/`
- `Figures/`
- `figures/`
- `latex_tables/`
- `human_validation_checks/`
- `human_validation_results/`

Generated outputs are intentionally not committed.

## Notes for manuscript reproduction

For Overleaf or manuscript compilation, copy the generated figures and LaTeX tables from the local output folders into the manuscript project. The repository stores the scripts needed to regenerate those artifacts, not the generated artifacts themselves.

## Code repository

This repository contains the code-only reproducibility pipeline for the manuscript:

- `https://github.com/mkboch/EDRPSS`
