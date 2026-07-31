# Sahyadri Spice Company — Pooled Testing

Submission for **Quantitative Techniques I, 2026-28**.

## Contents

| File | What it is |
|---|---|
| `app.py` | **Interactive Streamlit dashboard** — 8 tabs, every derivation and result |
| `sahyadri_pooling.py` | The model: closed forms, optimiser, Monte Carlo, imperfect-assay simulator |
| `run_analysis.py` | Reproduces every number and figure in the written report, non-interactively |
| `ANALYSIS.md` | The written recommendation for Rohan |
| `fig*.png`, `*.csv` | Static outputs from `run_analysis.py` |

## Running the dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

It opens at `http://localhost:8501`. `app.py` and `sahyadri_pooling.py` must be
in the same folder.

## Reproducing the static report

```bash
python run_analysis.py
```

Prints the validation table and all headline numbers, and writes the four
figures and three CSVs.

## Dashboard map

| Tab | Case question | Content |
|---|---|---|
| ① The models | setup | Both derivations, why the exponent is 2n−1, why 2×2 fails |
| ② Scheme A | Q1, Q2 | Dorfman optimum, cost curve, flat-band around the minimum |
| ③ Scheme B | Q3 | Grid derivation, small-p asymptotics, deduction refinement |
| ④ Head to head | Q3 | Cost comparison and the stage-1 / stage-2 split |
| ⑤ Robustness | Q4 | Sweep over p, the flip at 0.125, break-evens, minimax regret, estimating p |
| ⑥ Imperfect assay | Q5 | Dilution model, which error worsens, cost of ignoring vs fixing |
| ⑦ Validation | — | Simulation vs formula, the 1−3^(−1/3) benchmark, n ∤ N |
| ⑧ Recommendation | hand-in | The five-point recommendation |

The sidebar controls (contamination rate, N, cost per assay, replications)
drive every tab at once, so the whole analysis can be re-run live at any
contamination level.
