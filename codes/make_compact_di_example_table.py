from pathlib import Path
import pandas as pd
import re

ROOT = Path(".").resolve()
SRC = ROOT / "latex_tables" / "table_di_examples_source.csv"
OUT = ROOT / "latex_tables" / "table_di_examples.tex"

if not SRC.exists():
    raise FileNotFoundError(f"Missing source CSV: {SRC}")

df = pd.read_csv(SRC)

# Keep only 8 examples to avoid an oversized appendix table
df = df.head(8).copy()

def latex_escape(s):
    s = "" if pd.isna(s) else str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(c, c) for c in s)

def shorten_words(text, max_words=22):
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."

# Make interpretations less repetitive
default_interpretations = [
    "Host/tool execution language",
    "Access and integration language",
    "Safety checklist phrasing",
    "Credential/logging warning",
    "Supply-chain safety checklist",
    "Execution caution language",
    "Install/run command context",
    "Command-warning phrasing",
]

rows = []
for i, (_, r) in enumerate(df.iterrows(), start=1):
    di = int(r["DI"]) if "DI" in df.columns and not pd.isna(r["DI"]) else ""
    excerpt = shorten_words(r["excerpt"], 22)
    interp = default_interpretations[i-1] if i-1 < len(default_interpretations) else "Directive-marker-rich wording"
    rows.append((i, di, excerpt, interp))

with open(OUT, "w", encoding="utf-8") as f:
    f.write(r"""\begin{center}
\small
\begin{tabular}{@{}r c p{0.56\linewidth} p{0.24\linewidth}@{}}
\toprule
Ex. & DI & Short excerpt & What DI captures \\
\midrule
""")
    for ex, di, excerpt, interp in rows:
        f.write(
            f"{ex} & {di} & {latex_escape(excerpt)} & {latex_escape(interp)} \\\\\n"
        )
    f.write(r"""\bottomrule
\end{tabular}

\vspace{0.4em}
\parbox{0.92\linewidth}{\footnotesize
\textbf{Table A1. Illustrative high-DI post excerpts.}
Examples are shortened excerpts from posts with high directive-marker burden.
DI reflects matched directive-marker burden, not intent, harmfulness, or downstream execution.
Post identifiers are omitted here to keep the calibration table readable.
}
\end{center}
""")

print(f"Saved compact non-floating table: {OUT}")
print("Preview:")
for ex, di, excerpt, interp in rows:
    print(f"{ex}. DI={di} | {excerpt} | {interp}")
