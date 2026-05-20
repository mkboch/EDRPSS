from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(".").resolve()
OUT = ROOT / "journal_upgrade_final_tables"
OUT.mkdir(exist_ok=True)

ROUND2 = ROOT / "journal_upgrade_outputs_round2"
ROUND3 = ROOT / "journal_upgrade_outputs_round3"
ROUND4 = ROOT / "journal_upgrade_outputs_round4_auto_audit"

print("Project root:", ROOT)
print("Output folder:", OUT)

def fmt_p(p):
    try:
        p = float(p)
    except Exception:
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

def fmt_or_ci(row):
    try:
        return f"{float(row['odds_ratio']):.3f} [{float(row['or_ci_low']):.3f}, {float(row['or_ci_high']):.3f}]"
    except Exception:
        return ""

# ------------------------------------------------------------
# Table 1: Length-adjusted DI models
# ------------------------------------------------------------
r2_path = ROUND2 / "round2_length_normalized_di_models.csv"
r3_path = ROUND3 / "round3_binary_di_models.csv"

r2 = pd.read_csv(r2_path)
r3 = pd.read_csv(r3_path)

main_terms = [
    "z_di_post",
    "z_di_per_100_words",
    "z_di_per_1000_chars",
    "z_di_log_density_words",
    "z_di_resid_length",
]

r2_main = r2[r2["term"].isin(main_terms)].copy()
r3_main = r3[r3["term"].eq("di_binary")].copy()

table1_rows = []

model_name_map = {
    "raw_DI_no_controls": "Raw DI, no controls",
    "raw_DI_plus_length": "Raw DI + post length",
    "raw_DI_plus_length_thread_score": "Raw DI + length + thread size + score",
    "DI_per_100_words_plus_length_thread_score": "DI per 100 words + controls",
    "DI_per_1000_chars_plus_length_thread_score": "DI per 1000 chars + controls",
    "DI_log_density_words_plus_length_thread_score": "Log DI density + controls",
    "DI_resid_length_plus_thread_score": "Length-residualized DI + controls",
    "binary_DI_no_controls": "Binary DI, no controls",
    "binary_DI_plus_post_length_chars": "Binary DI + post length",
    "binary_DI_plus_length_thread_score": "Binary DI + length + thread size + score",
}

keep_models = [
    "raw_DI_no_controls",
    "raw_DI_plus_length",
    "raw_DI_plus_length_thread_score",
    "DI_per_100_words_plus_length_thread_score",
    "DI_per_1000_chars_plus_length_thread_score",
    "DI_log_density_words_plus_length_thread_score",
    "DI_resid_length_plus_thread_score",
]

for model in keep_models:
    sub = r2_main[r2_main["model"].eq(model)]
    if len(sub) == 0:
        continue
    row = sub.iloc[0]
    table1_rows.append({
        "Model": model_name_map.get(model, model),
        "DI term": row["term"],
        "OR [95% CI]": fmt_or_ci(row),
        "p-value": fmt_p(row["p_value"]),
        "N": int(row["n_used"]) if "n_used" in row else "",
        "Post clusters": int(row["n_clusters"]) if "n_clusters" in row else "",
    })

for model in ["binary_DI_no_controls", "binary_DI_plus_post_length_chars", "binary_DI_plus_length_thread_score"]:
    sub = r3_main[r3_main["model"].eq(model)]
    if len(sub) == 0:
        continue
    row = sub.iloc[0]
    table1_rows.append({
        "Model": model_name_map.get(model, model),
        "DI term": row["term"],
        "OR [95% CI]": fmt_or_ci(row),
        "p-value": fmt_p(row["p_value"]),
        "N": int(row["n_used"]) if "n_used" in row else "",
        "Post clusters": int(row["n_clusters"]) if "n_clusters" in row else "",
    })

table1 = pd.DataFrame(table1_rows)
table1.to_csv(OUT / "table1_length_adjusted_di_models.csv", index=False)
table1.to_latex(
    OUT / "table1_length_adjusted_di_models.tex",
    index=False,
    escape=False,
    caption="Length-adjusted and length-normalized DI models predicting corrective replies.",
    label="tab:length_adjusted_di_models"
)

# ------------------------------------------------------------
# Table 2: Automatic label audit summary
# ------------------------------------------------------------
risk_path = ROUND4 / "round4_auto_label_quality_risk_summary.csv"
risk = pd.read_csv(risk_path)

table2 = risk.copy()
table2["Expected pattern present"] = (100 * table2["pct_expected_pattern_present"]).map(lambda x: f"{x:.1f}%")
table2["Expected pattern absent"] = (100 * table2["pct_expected_pattern_absent"]).map(lambda x: f"{x:.1f}%")
table2["Multiple pattern groups"] = (100 * table2["pct_multiple_pattern_groups"]).map(lambda x: f"{x:.1f}%")
table2["No audit pattern"] = (100 * table2["pct_no_pattern_matched"]).map(lambda x: f"{x:.1f}%")

table2 = table2.rename(columns={
    "response_type": "Response type",
    "n": "N",
    "median_comment_words": "Median words",
})

table2 = table2[[
    "Response type",
    "N",
    "Expected pattern present",
    "Expected pattern absent",
    "Multiple pattern groups",
    "No audit pattern",
    "Median words",
]]

table2.to_csv(OUT / "table2_auto_label_audit_summary.csv", index=False)
table2.to_latex(
    OUT / "table2_auto_label_audit_summary.tex",
    index=False,
    escape=False,
    caption="Automatic lexical audit of deterministic response-type labels.",
    label="tab:auto_label_audit"
)

# ------------------------------------------------------------
# Table 3: Candidate examples for paper
# ------------------------------------------------------------
examples_path = ROUND4 / "round4_candidate_examples_for_paper_table.csv"
examples = pd.read_csv(examples_path)

# Keep two per response type for compact paper table
example_rows = []
for label in ["corrective", "affirmation", "adversarial", "neutral"]:
    sub = examples[examples["response_type"].eq(label)].copy()
    if len(sub) == 0:
        continue
    sub["comment_len"] = sub["comment_excerpt"].fillna("").astype(str).str.len()
    sub = sub.sort_values("comment_len")
    for _, row in sub.head(2).iterrows():
        example_rows.append({
            "Response type": row["response_type"],
            "Matched family": row.get("matched_rule_family", ""),
            "Matched phrase": row.get("matched_phrase", ""),
            "Comment excerpt": str(row.get("comment_excerpt", ""))[:220],
        })

table3 = pd.DataFrame(example_rows)
table3.to_csv(OUT / "table3_candidate_label_examples.csv", index=False)
table3.to_latex(
    OUT / "table3_candidate_label_examples.tex",
    index=False,
    escape=True,
    caption="Candidate examples from the automatic label audit.",
    label="tab:label_examples"
)

# ------------------------------------------------------------
# Short interpretation text
# ------------------------------------------------------------
text = r"""
\paragraph{Length-adjusted interpretation.}
The unadjusted raw DI model reproduces the positive association between directive-marker burden and corrective replies.
However, this association is not stable after controlling for post length.
Raw DI reverses direction after length adjustment, while length-normalized and length-residualized DI variants provide weak or inconsistent evidence of an independent directive-density effect.
The binary DI specification shows the same pattern: posts containing at least one directive marker are more likely to receive corrective replies in the unadjusted model, but this association is not significant after post-length adjustment.
These results indicate that DI should be interpreted as a raw directive-marker burden measure that is strongly entangled with post length, rather than as evidence that directive density alone predicts correction.

\paragraph{Automatic label-audit interpretation.}
The automatic lexical audit provides a post-hoc transparency check for deterministic response labels.
Corrective and affirmation labels show relatively high expected-pattern coverage, while adversarial labels show weak coverage under the audit patterns used here.
The high rate of multiple pattern-group matches also indicates that many comments contain overlapping lexical cues.
Therefore, response labels should be interpreted as rule-derived operational categories rather than ground-truth pragmatic judgments.
This audit supports measurement transparency but does not replace human semantic validation.
"""
(OUT / "paper_interpretation_text.tex").write_text(text.strip(), encoding="utf-8")

print("\nCreated final table files:")
for p in sorted(OUT.glob("*")):
    print(" -", p)

print("\nTable 1 preview:")
print(table1.to_string(index=False))

print("\nTable 2 preview:")
print(table2.to_string(index=False))

print("\nDone.")
