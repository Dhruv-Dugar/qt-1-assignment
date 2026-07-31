"""Runs the full analysis for the Sahyadri case and writes figures."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sahyadri_pooling import (row_pooling_per_lot, grid_per_lot, grid_per_lot_deduced,
                              optimise, cost_curve, simulate,
                              row_pooling_per_lot_imperfect, grid_per_lot_imperfect,
                              escape_rate, condemn_rate)

N = 4000
# Write outputs next to this script, unless OUT_DIR says otherwise.
OUT = os.environ.get("OUT_DIR", os.path.dirname(os.path.abspath(__file__)) or ".")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3})

# ---------------------------------------------------------------- 0. VALIDATE
print("=" * 70)
print("VALIDATION: closed form vs Monte Carlo (N=4000, 2000 reps)")
print("=" * 70)
rows = []
for scheme, fn in [("row_pooling", row_pooling_per_lot), ("grid", grid_per_lot)]:
    for p in (0.05, 0.10, 0.20):
        for n in (4, 6, 9):
            a = fn(n, p)
            m, se = simulate(scheme, n, p, N=N, reps=2000, seed=42)
            rows.append(dict(scheme=scheme, p=p, n=n, analytic=round(a, 5),
                             simulated=round(m, 5), se=round(se, 5),
                             z=round((m - a) / se, 2)))
val = pd.DataFrame(rows)
print(val.to_string(index=False))
val.to_csv(f"{OUT}/validation_table.csv", index=False)

# ---------------------------------------------- 1&2. ROW POOLING, TYPICAL YEAR
print("\n" + "=" * 70)
print("Q2  Row Pooling at p = 0.05")
print("=" * 70)
p0 = 0.05
nD, cD = optimise(row_pooling_per_lot, p0)
print(f"optimal group size n* = {nD}, cost/lot = {cD:.4f}")
print(f"total assays = {cD*N:,.0f} vs {N:,} individual  "
      f"({100*(1-cD):.1f}% saving)")
ns, cs = cost_curve(row_pooling_per_lot, p0, 2, 30)
for n in range(3, 9):
    print(f"   n={n:2d}  cost/lot={row_pooling_per_lot(n,p0):.4f}  "
          f"total={row_pooling_per_lot(n,p0)*N:,.0f}")

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(ns, cs * N, "o-", ms=3.5, label="Row Pooling")
ax.axhline(N, ls="--", c="crimson", label="test every lot (4000)")
ax.plot(nD, cD * N, "*", ms=15, c="darkgreen", zorder=5,
        label=f"optimum n={nD}: {cD*N:,.0f} assays")
ax.set_xlabel("group size n"); ax.set_ylabel("expected assays for 4000 lots")
ax.set_title(f"Scheme A: expected cost vs group size (p={p0})")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(f"{OUT}/fig1_row_pooling_p05.png"); plt.close(fig)

# ------------------------------------------------------ 3. GRID, TYPICAL YEAR
print("\n" + "=" * 70)
print("Q3  Cross pooling at p = 0.05, head-to-head")
print("=" * 70)
nG, cG = optimise(grid_per_lot, p0)
nGd, cGd = optimise(grid_per_lot_deduced, p0)
print(f"grid side n* = {nG}, cost/lot = {cG:.4f}, total = {cG*N:,.0f}")
print(f"  (with the 'single positive is forced' deduction: n*={nGd}, "
      f"{cGd:.4f}/lot, {cGd*N:,.0f})")
print(f"grid beats Row Pooling by {100*(cD-cG)/cD:.1f}% of Row Pooling's cost")
for n in range(5, 13):
    print(f"   n={n:2d}  cost/lot={grid_per_lot(n,p0):.4f}  "
          f"total={grid_per_lot(n,p0)*N:,.0f}")
print(f"   n=2 (Task#2 in notes): cost/lot = {grid_per_lot(2,p0):.4f} "
      f"-> {'WORSE' if grid_per_lot(2,p0)>=1 else 'ok'} than individual testing")

ns2, cs2 = cost_curve(grid_per_lot, p0, 2, 30)
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(ns, cs, "o-", ms=3.5, label=f"Scheme A: Row Pooling (min {cD:.3f} at n={nD})")
ax.plot(ns2, cs2, "s-", ms=3.5, label=f"Scheme B grid (min {cG:.3f} at n={nG})")
ax.axhline(1.0, ls="--", c="crimson", label="test every lot")
ax.set_xlabel("pool size / grid side n"); ax.set_ylabel("expected assays per lot")
ax.set_title(f"Head to head at p={p0}"); ax.set_ylim(0.3, 1.15)
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(f"{OUT}/fig2_head_to_head_p05.png"); plt.close(fig)

# ------------------------------------------------------ 4. ROBUSTNESS SWEEP
print("\n" + "=" * 70)
print("Q4  Sweep over contamination rate")
print("=" * 70)
ps = np.round(np.arange(0.03, 0.351, 0.005), 4)
rec = []
for p in ps:
    a_n, a_c = optimise(row_pooling_per_lot, p)
    b_n, b_c = optimise(grid_per_lot, p)
    rec.append(dict(p=p, row_pooling_n=a_n, row_pooling_cost=a_c,
                    grid_n=b_n, grid_cost=b_c,
                    winner="grid" if b_c < a_c else "row_pooling",
                    best_cost=min(a_c, b_c)))
sw = pd.DataFrame(rec)
sw.to_csv(f"{OUT}/sweep_by_p.csv", index=False)
show = sw[sw.p.isin([0.05, 0.08, 0.10, 0.12, 0.125, 0.13, 0.15, 0.20, 0.25, 0.30, 0.31])]
print(show.round(4).to_string(index=False))

flip = sw[sw.winner == "row_pooling"].p.min()
d_break = sw[sw.row_pooling_cost >= 1.0].p.min()
g_break = sw[sw.grid_cost >= 1.0].p.min()
print(f"\nranking flips (grid -> Row Pooling) at p ~ {flip:.3f}")
print(f"grid stops beating individual testing at p ~ {g_break:.3f}")
print(f"Row Pooling stops beating individual testing at p ~ {d_break:.3f} "
      f"(theory: 1 - 3^(-1/3) = {1-3**(-1/3):.4f})")

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
ax = axes[0]
ax.plot(sw.p, sw.row_pooling_cost, "-", lw=2, label="Scheme A: Row Pooling")
ax.plot(sw.p, sw.grid_cost, "-", lw=2, label="Scheme B: grid")
ax.axhline(1.0, ls="--", c="crimson", label="test every lot")
ax.axvline(flip, ls=":", c="k")
ax.annotate(f"ranking flips\np≈{flip:.3f}", (flip, 0.45), fontsize=8,
            xytext=(flip + 0.02, 0.42))
ax.axvspan(0.05, 0.20, color="grey", alpha=0.12)
ax.set_xlabel("contaminated fraction p")
ax.set_ylabel("best achievable assays per lot")
ax.set_title("Optimised cost per lot (shaded = Rohan's stated range)")
ax.legend(fontsize=8)
ax = axes[1]
ax.step(sw.p, sw.row_pooling_n, where="mid", lw=2, label="Row Pooling group size n*")
ax.step(sw.p, sw.grid_n, where="mid", lw=2, label="grid side n*")
ax.set_xlabel("contaminated fraction p"); ax.set_ylabel("optimal n")
ax.set_ylim(2, 12)   # grid n* degenerates >0.25 where the scheme is useless anyway
ax.set_title("Optimal size shrinks as contamination rises")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_robustness.png"); plt.close(fig)

# fixed-n robustness: what if you pick one size and the season surprises you
print("\nRobustness of a FIXED choice across p in [0.05, 0.20]:")
grid_p = np.arange(0.05, 0.2001, 0.005)
for label, fn, cands in [("Row Pooling", row_pooling_per_lot, range(3, 9)),
                         ("grid", grid_per_lot, range(4, 12))]:
    for n in cands:
        worst = max(fn(n, p) for p in grid_p)
        regret = max(fn(n, p) - min(optimise(row_pooling_per_lot, p)[1],
                                    optimise(grid_per_lot, p)[1]) for p in grid_p)
        print(f"  {label:11s} n={n:2d}  worst-case cost/lot={worst:.4f}  "
              f"max regret vs oracle={regret:.4f}")

# --------------------------------------------------------- 5. IMPERFECT ASSAY
print("\n" + "=" * 70)
print("Q5  Imperfect assay, gate model (Se=0.98, Sp=0.99), p=0.05")
print("=" * 70)
Se, Sp = 0.98, 0.99
res = []
for label, k, n in [("individual", 1, 1),
                    (f"Row Pooling n={nD}", 2, nD),
                    (f"grid {nG}x{nG}", 3, nG)]:
    if k == 1:
        assays = float(N)
    elif k == 2:
        assays = row_pooling_per_lot_imperfect(n, p0, Se, Sp) * N
    else:
        assays = grid_per_lot_imperfect(n, p0, Se, Sp) * N
    esc = escape_rate(k, Se)
    cond = condemn_rate(k, n, p0, Se, Sp)
    res.append(dict(scheme=label, gates=k, assays_per_lot=assays / N,
                    escape_rate=esc, escaped_per_season=esc * p0 * N,
                    condemned_per_season=cond * (1 - p0) * N))
imp = pd.DataFrame(res)
print(imp.round(4).to_string(index=False))
imp.to_csv(f"{OUT}/imperfect_assay.csv", index=False)
print("\ncsv written to", OUT)
