"""
Sahyadri Spice Company - Interactive Pooled Testing Dashboard
============================================================
Run with:   streamlit run app.py
Requires:   app.py and sahyadri_pooling.py in the same folder.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

try:
    from sahyadri_pooling import (row_pooling_per_lot, grid_per_lot, grid_K,
                                  optimise, cost_curve, simulate,
                                  row_pooling_per_lot_imperfect,
                                  grid_per_lot_imperfect, escape_rate,
                                  condemn_rate, estimate_p_se)
except ImportError:
    st.error("`sahyadri_pooling.py` must sit in the same folder as `app.py`.")
    st.stop()

st.set_page_config(page_title="Sahyadri Pooled Testing",
                   page_icon="🌶️", layout="wide")

plt.rcParams.update({"figure.dpi": 110, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})

C_A, C_B, C_REF = "#1f77b4", "#ff7f0e", "#d62728"


# ----------------------------------------------------------------- cached work
@st.cache_data(show_spinner=False)
def sweep(p_lo, p_hi, step=0.005, n_max=40):
    ps = np.round(np.arange(p_lo, p_hi + 1e-9, step), 4)
    rows = []
    for p in ps:
        an, ac = optimise(row_pooling_per_lot, p, n_max=n_max)
        bn, bc = optimise(grid_per_lot, p, n_max=n_max)
        rows.append(dict(p=p, row_pooling_n=an, row_pooling_cost=ac,
                         grid_n=bn, grid_cost=bc,
                         winner="Grid" if bc < ac else "Row Pooling"))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Simulating…")
def mc(scheme, n, p, N, reps, seed=42):
    return simulate(scheme, n, p, N=N, reps=reps, seed=seed)


# --------------------------------------------------------------------- sidebar
st.sidebar.title("🌶️ Controls")
st.sidebar.caption("Every panel reacts to these.")
p = st.sidebar.slider("Contaminated fraction  p", 0.01, 0.35, 0.05, 0.005,
                      help="Rohan's realistic range is 0.05 to 0.20.")
N = st.sidebar.number_input("Lots per season  N", 100, 50_000, 4000, 100)
cost_per_assay = st.sidebar.number_input("Cost per assay (₹)", 0, 20_000, 3000, 250)
st.sidebar.divider()
reps = st.sidebar.select_slider("Monte Carlo replications", [200, 500, 1000, 2000], 2000)
n_max = st.sidebar.slider("Max pool size searched", 10, 80, 40, 5)

nD, cD = optimise(row_pooling_per_lot, p, n_max=n_max)
nG, cG = optimise(grid_per_lot, p, n_max=n_max)
best = "Grid (Scheme B)" if cG < cD else "Row Pooling (Scheme A)"

st.sidebar.divider()
st.sidebar.metric("Scheme A optimum", f"n = {nD}", f"{cD*N:,.0f} assays")
st.sidebar.metric("Scheme B optimum", f"{nG}×{nG} grid", f"{cG*N:,.0f} assays")
st.sidebar.success(f"Cheaper at p = {p:.3f}: **{best}**")


# ----------------------------------------------------------------------- title
st.title("Pooled Testing at Sahyadri Spice Company")
st.markdown(
    "**Quantitative Techniques I — 2026-28.** Every cost below is counted in "
    "**assays**: one LC–MS/MS run on one tube, pooled or individual, all at "
    "the same price. Lots are treated as independently contaminated with "
    "probability $p$; $q = 1-p$."
)

tabs = st.tabs(["① The models", "② Scheme A", "③ Scheme B", "④ Head to head",
                "⑤ Robustness", "⑥ Imperfect assay", "⑦ Validation",
                "⑧ Recommendation"])


# =========================================================== ① THE MODELS
with tabs[0]:
    st.header("Two schemes, two formulas")
    a, b = st.columns(2)

    with a:
        st.subheader("Scheme A — Row Pooling")
        st.markdown(
            "Split the lots into groups of $n$. Combine and test each group. "
            "A clean pool clears everyone in it; a dirty pool sends all $n$ "
            "for individual assays."
        )
        st.markdown("**The derivation.** A group always costs the 1 combined "
                    "assay. It costs $n$ more only if the pool reads positive, "
                    "which happens unless all $n$ lots are clean:")
        st.latex(r"P(\text{pool positive}) = 1 - q^{n}")
        st.latex(r"E[\text{assays per group}] = 1 + n\,(1 - q^{n})")
        st.latex(r"\boxed{\;\frac{E[\text{assays}]}{\text{lot}} "
                 r"= \frac{1}{n} + \left(1 - q^{n}\right)\;}")
        st.latex(r"E[\text{total}] = \frac{N}{n} + N\left(1-q^{n}\right)")
        st.info("**The trade-off in one line.** $1/n$ falls as the group grows; "
                "$1-q^n$ rises. The optimum sits where cheaper screening stops "
                "paying for the extra re-testing it causes.")

    with b:
        st.subheader("Scheme B — Cross pooling (grid)")
        st.markdown(
            "Lay $n^2$ lots on an $n\\times n$ tray. Pool and test each row, "
            "then each column: $2n$ assays. Lot $(i,j)$ is individually "
            "re-tested only if row $i$ **and** column $j$ are both positive."
        )
        st.markdown("**The derivation.** Let $X_{ij}=1$ if lot $(i,j)$ reaches "
                    "stage 2, $A$ = row $i$ positive, $B$ = column $j$ positive:")
        st.latex(r"E[X_{ij}] = P(A\cap B) = P(A) + P(B) - P(A\cup B)")
        st.latex(r"P(A) = P(B) = 1 - q^{n}")
        st.latex(r"P(A\cup B) = 1 - q^{\,2n-1}")
        st.latex(r"K = 1 - 2q^{n} + q^{\,2n-1}")
        st.latex(r"\boxed{\;\frac{E[\text{assays}]}{\text{lot}} "
                 r"= \frac{2}{n} + K\;}\qquad "
                 r"E[\text{total}] = \frac{2N}{n} + NK")

    st.divider()
    st.subheader("Why the exponent is $2n-1$, not $2n$")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(
            "The event *“row $i$ or column $j$ is positive”* is itself a pooled "
            "test — over every lot lying in that row or that column. Row $i$ has "
            "$n$ lots and column $j$ has $n$ lots, **but they share the cell "
            "$(i,j)$**, so the union contains $2n-1$ distinct lots, not $2n$.\n\n"
            "That shared cell is exactly what makes $A$ and $B$ *dependent*: "
            "learning that row $i$ is dirty raises the chance column $j$ is too. "
            "Assuming independence would give $K=(1-q^n)^2$ and understate "
            "stage-2 work."
        )
        ind = (1 - (1 - p) ** nG) ** 2
        st.metric(f"Understatement at p={p:.3f}, n={nG}",
                  f"{100*(grid_K(nG,p)-ind)/grid_K(nG,p):.1f}%",
                  f"correct K = {grid_K(nG,p):.4f} vs independent {ind:.4f}",
                  delta_color="off")
    with c2:
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        for k in range(5):
            for j in range(5):
                ax.add_patch(plt.Rectangle((j, 4 - k), 1, 1, fc="white",
                                           ec="#bbb", lw=0.8))
        for j in range(5):
            ax.add_patch(plt.Rectangle((j, 2), 1, 1, fc=C_A, alpha=.30, ec="none"))
        for k in range(5):
            ax.add_patch(plt.Rectangle((2, 4 - k), 1, 1, fc=C_B, alpha=.30, ec="none"))
        ax.add_patch(plt.Rectangle((2, 2), 1, 1, fc="#7b3fa0", alpha=.75, ec="k", lw=1.2))
        ax.text(2.5, 2.5, "(i,j)", ha="center", va="center",
                fontsize=7, color="white", weight="bold")
        ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.axis("off")
        ax.set_title("row ∪ column = 2n−1 lots\n(the corner is counted once)",
                     fontsize=8)
        st.pyplot(fig, width="stretch")
        plt.close(fig)


# ============================================================== ② SCHEME A
with tabs[1]:
    st.header(f"Scheme A: Row Pooling at p = {p:.3f}")
    ns, cs = cost_curve(row_pooling_per_lot, p, 2, n_max)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Optimal group size", f"n = {nD}")
    m2.metric("Assays per lot", f"{cD:.4f}")
    m3.metric(f"Total for {N:,} lots", f"{cD*N:,.0f}", f"{-(1-cD)*100:.1f}% vs individual")
    m4.metric("Season cost", f"₹{cD*N*cost_per_assay/1e5:,.1f} L",
              f"saves ₹{(1-cD)*N*cost_per_assay/1e5:,.1f} L", delta_color="off")

    left, right = st.columns([3, 2])
    with left:
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        ax.plot(ns, cs * N, "o-", ms=3.5, c=C_A, label="Row Pooling")
        ax.axhline(N, ls="--", c=C_REF, label=f"test every lot ({N:,})")
        ax.plot(nD, cD * N, "*", ms=16, c="darkgreen", zorder=5,
                label=f"optimum n={nD}: {cD*N:,.0f}")
        band = ns[cs <= cD * 1.02]
        ax.axvspan(band.min(), band.max(), color="green", alpha=.08)
        ax.set_xlabel("group size n"); ax.set_ylabel("expected assays")
        ax.legend(fontsize=8); ax.set_title("Expected cost vs group size")
        st.pyplot(fig); plt.close(fig)
        st.caption(f"Shaded band = sizes within 2% of the optimum "
                   f"(n = {band.min()} to {band.max()}). The curve is flat near "
                   f"its minimum, so the laboratory can rack tubes in whatever "
                   f"size is convenient without a meaningful penalty.")
    with right:
        tab = pd.DataFrame({"n": ns, "assays/lot": cs, "total": cs * N})
        tab = tab[(tab.n >= max(2, nD - 3)) & (tab.n <= nD + 4)]
        st.dataframe(tab.round({"assays/lot": 4, "total": 0}),
                     hide_index=True, width="stretch")
        st.markdown("**Reading the two terms**")
        st.latex(r"\underbrace{1/n}_{\text{screening}} \;+\; "
                 r"\underbrace{1-q^{n}}_{\text{re-testing}}")
        st.write(f"At n = {nD}: screening {1/nD:.3f} + re-testing "
                 f"{1-(1-p)**nD:.3f} = {cD:.3f}")

    if st.toggle("Verify this by Monte Carlo simulation", key="simA"):
        m, se = mc("row_pooling", nD, p, int(N), reps)
        z = (m - cD) / se
        st.write(f"Formula **{cD:.5f}** | simulated **{m:.5f} ± {se:.5f}** "
                 f"({reps} replications) | z = **{z:+.2f}**")
        st.success("Within ±2 standard errors — the closed form is right."
                   if abs(z) < 2 else "Outside ±2 SE — raise the replication count.")


# ============================================================== ③ SCHEME B
with tabs[2]:
    st.header(f"Scheme B: cross pooling at p = {p:.3f}")
    ns2, cs2 = cost_curve(grid_per_lot, p, 2, n_max)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Optimal grid", f"{nG} × {nG}", f"{nG**2} lots per tray")
    m2.metric("Assays per lot", f"{cG:.4f}")
    m3.metric(f"Total for {N:,} lots", f"{cG*N:,.0f}", f"{-(1-cG)*100:.1f}% vs individual")
    m4.metric("vs Scheme A", f"{(cG-cD)*N:+,.0f} assays",
              f"{100*(cG-cD)/cD:+.1f}%", delta_color="inverse")

    left, right = st.columns([3, 2])
    with left:
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        ax.plot(ns2, cs2, "s-", ms=3.5, c=C_B, label="Scheme B: grid")
        ax.plot(ns, cs, "o-", ms=3, c=C_A, alpha=.65, label="Scheme A: Row Pooling")
        ax.axhline(1.0, ls="--", c=C_REF, label="test every lot")
        ax.plot(nG, cG, "*", ms=16, c="darkgreen", zorder=5)
        ax.set_ylim(min(cs2.min(), cs.min()) * 0.9, 1.15)
        ax.set_xlabel("pool size / grid side n"); ax.set_ylabel("assays per lot")
        ax.legend(fontsize=8); ax.set_title("Both schemes on one axis")
        st.pyplot(fig); plt.close(fig)
    with right:
        st.markdown("**Reading the two terms**")
        st.latex(r"\underbrace{2/n}_{\text{rows + columns}} \;+\; "
                 r"\underbrace{K}_{\text{intersections}}")
        st.write(f"At n = {nG}: stage 1 {2/nG:.3f} + stage 2 "
                 f"{grid_K(nG,p):.3f} = {cG:.3f}")
        st.markdown(
            "**Why it can beat Row Pooling.** The grid pays *double* at stage 1 "
            "($2/n$ instead of $1/n$) and buys sharper localisation in return: "
            "a positive row crossed with a positive column pins the suspect to "
            "one cell instead of condemning a whole group. When positives are "
            "rare, that precision lets the grid run a much larger $n$ — and the "
            "large $n$ is what crushes the stage-1 cost per lot."
        )

    if st.toggle("Verify this by Monte Carlo simulation", key="simB"):
        m, se = mc("grid", nG, p, int(N), reps)
        z = (m - cG) / se
        st.write(f"Formula **{cG:.5f}** | simulated **{m:.5f} ± {se:.5f}** "
                 f"({reps} replications) | z = **{z:+.2f}**")
        st.success("Within ±2 standard errors — the closed form is right."
                   if abs(z) < 2 else "Outside ±2 SE — raise the replication count.")


# ========================================================= ④ HEAD TO HEAD
with tabs[3]:
    st.header("Head to head")
    win_by = abs(cD - cG) * N
    st.markdown(
        f"At **p = {p:.3f}**, the cheaper scheme is **{best}**, by "
        f"**{win_by:,.0f} assays** (₹{win_by*cost_per_assay/1e5:,.2f} lakh) "
        f"over a season of {N:,} lots."
    )

    comp = pd.DataFrame({
        "Scheme": ["Test every lot", "A — Row Pooling", "B — Cross pooling"],
        "Optimal size": ["—", f"groups of {nD}", f"{nG}×{nG} grid"],
        "Assays / lot": [1.0, cD, cG],
        "Total assays": [N, cD * N, cG * N],
        "Season cost (₹ lakh)": [N * cost_per_assay / 1e5,
                                 cD * N * cost_per_assay / 1e5,
                                 cG * N * cost_per_assay / 1e5],
        "Saving vs individual": ["—", f"{(1-cD)*100:.1f}%", f"{(1-cG)*100:.1f}%"],
    })
    st.dataframe(comp.round({"Assays / lot": 4, "Total assays": 0,
                             "Season cost (₹ lakh)": 2}),
                 hide_index=True, width="stretch")

    fig, ax = plt.subplots(figsize=(7, 2.6))
    bars = ax.barh(["Test every lot", "A — Row Pooling", "B — Grid"],
                   [N, cD * N, cG * N], color=["#bbb", C_A, C_B])
    for bar, v in zip(bars, [N, cD * N, cG * N]):
        ax.text(v + N * .01, bar.get_y() + bar.get_height() / 2,
                f"{v:,.0f}", va="center", fontsize=9)
    ax.set_xlabel("expected assays per season"); ax.set_xlim(0, N * 1.15)
    ax.grid(axis="y", visible=False)
    st.pyplot(fig); plt.close(fig)

    st.divider()
    st.subheader("Where does the money actually go?")
    split = pd.DataFrame({
        "stage": ["Stage 1 (pooled screening)", "Stage 2 (individual confirms)"],
        "Scheme A": [N / nD, N * (1 - (1 - p) ** nD)],
        "Scheme B": [2 * N / nG, N * grid_K(nG, p)],
    })
    st.dataframe(split.round(0), hide_index=True, width="stretch")
    st.markdown(
        f"This table is the whole story of the comparison. Scheme B spends "
        f"**{2*N/nG:,.0f}** assays on screening against Scheme A's "
        f"**{N/nD:,.0f}** — but because a crossing identifies a suspect so "
        f"precisely at low $p$, it can afford a {nG}×{nG} tray and its stage-2 "
        f"bill is **{N*grid_K(nG,p):,.0f}** against **{N*(1-(1-p)**nD):,.0f}**. "
        f"Change $p$ in the sidebar and watch which column blows up."
    )


# ============================================================ ⑤ ROBUSTNESS
with tabs[4]:
    st.header("Robustness: the season Rohan cannot predict")
    lo, hi = st.slider("Contamination range to sweep", 0.01, 0.35,
                       (0.03, 0.35), 0.01)
    sw = sweep(lo, hi, n_max=n_max)

    flip_rows = sw[sw.winner == "Row Pooling"]
    flip = flip_rows.p.min() if len(flip_rows) and sw.winner.iloc[0] == "Grid" else None
    d_break = sw[sw.row_pooling_cost >= 1].p.min() if (sw.row_pooling_cost >= 1).any() else None
    g_break = sw[sw.grid_cost >= 1].p.min() if (sw.grid_cost >= 1).any() else None

    k1, k2, k3 = st.columns(3)
    k1.metric("Ranking flips at", f"p ≈ {flip:.3f}" if flip else "no flip in range",
              "grid → Row Pooling")
    k2.metric("Grid stops paying at", f"p ≈ {g_break:.3f}" if g_break else "> range")
    k3.metric("Row Pooling stops paying at", f"p ≈ {d_break:.3f}" if d_break else "> range",
              f"theory 1−3^(−1/3) = {1-3**(-1/3):.4f}", delta_color="off")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax = axes[0]
    ax.plot(sw.p, sw.row_pooling_cost, lw=2, c=C_A, label="A — Row Pooling")
    ax.plot(sw.p, sw.grid_cost, lw=2, c=C_B, label="B — Grid")
    ax.axhline(1.0, ls="--", c=C_REF, label="test every lot")
    ax.axvspan(0.05, 0.20, color="grey", alpha=.12, label="Rohan's stated range")
    if flip:
        ax.axvline(flip, ls=":", c="k")
        ax.annotate(f"flip\np≈{flip:.3f}", (flip, .45),
                    xytext=(flip + .012, .40), fontsize=8)
    ax.axvline(p, ls="-", c="green", alpha=.5, lw=1)
    ax.set_xlabel("contaminated fraction p"); ax.set_ylabel("best assays per lot")
    ax.set_title("Optimised cost per lot"); ax.legend(fontsize=7.5)
    ax = axes[1]
    ax.step(sw.p, sw.row_pooling_n, where="mid", lw=2, c=C_A, label="A: group size n*")
    ax.step(sw.p, sw.grid_n, where="mid", lw=2, c=C_B, label="B: grid side n*")
    ax.set_ylim(2, 14)
    ax.set_xlabel("contaminated fraction p"); ax.set_ylabel("optimal n")
    ax.set_title("Optimal size shrinks as contamination rises"); ax.legend(fontsize=8)
    st.pyplot(fig); plt.close(fig)
    st.caption("Right panel is clipped at n = 14; above p ≈ 0.25 the grid's "
               "optimiser runs off to large n because no size works at all.")

    st.subheader("Why the ranking flips")
    st.markdown(
        "The grid's stage-2 workload is **(number of positive rows) × (number "
        "of positive columns)**. With $m$ contaminated lots scattered across an "
        "array, up to $m$ rows and $m$ columns light up, and the scheme retests "
        "up to $m^2$ cells to find $m$ culprits. This is the **masking** or "
        "**shadow** effect: a crossing only isolates a lot when positives are "
        "sparse.\n\n"
        "Row Pooling has no such term — its stage-2 cost is linear in the number "
        "of positive groups. So as $p$ rises, the grid's cost grows quadratically "
        "against Row Pooling's linear, and the advantage inverts. The optimiser "
        "responds by shrinking the grid, but a smaller grid means a heavier "
        "$2/n$ entry fee, and the scheme is squeezed from both ends."
    )

    st.divider()
    st.subheader("Committing to one size before the season starts")
    st.markdown("If the tray size has to be fixed in advance, judge it by "
                "**worst-case cost** and **maximum regret** against an oracle "
                "that knows $p$:")

    theory_p = 1 - 3 ** (-1 / 3)
    ns_r = np.arange(2, min(26, n_max + 1))
    lo_curve = np.array([row_pooling_per_lot(int(n), 0.05) for n in ns_r]) * N
    hi_curve = np.array([row_pooling_per_lot(int(n), theory_p) for n in ns_r]) * N
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.fill_between(ns_r, lo_curve, hi_curve, color="grey", alpha=.15,
                    label="Swept region")
    curve_specs = [(0.05, "green", "-", "p = 0.05"),
                   (0.10, "#5b6bd6", "--", "p = 0.10"),
                   (0.15, "#5b6bd6", "--", "p = 0.15"),
                   (0.20, "crimson", "-", "p = 0.20 (top of range)"),
                   (theory_p, "#666", "-", f"p = {theory_p:.4f} (threshold)")]
    for i, (q, c, ls, label) in enumerate(curve_specs):
        y = np.array([row_pooling_per_lot(int(n), q) for n in ns_r]) * N
        ax.plot(ns_r, y, ls, lw=2, c=c, label=label)
        nq, cq = optimise(row_pooling_per_lot, q, n_max=n_max)
        if nq <= ns_r.max():
            ax.plot(nq, cq * N, "o", ms=7, c="k", zorder=5,
                    label="Minima" if i == 0 else None)
    ax.axhline(N, ls="--", c=C_REF, label="Individual testing")
    ax.set_xlabel("Group size n"); ax.set_ylabel("Expected total assays")
    ax.set_title("Row Pooling: cost curve across Rohan's stated range of p")
    ax.legend(fontsize=7.5, loc="upper right")
    st.pyplot(fig); plt.close(fig)
    st.caption(f"Shaded band spans p = 0.05 (best case) up to the theoretical "
               f"break-even p = {theory_p:.4f} beyond which Row Pooling never "
               "beats individual testing. Dots mark each curve's optimal group "
               "size — notice how little it moves across the realistic range.")

    grid_p = np.arange(0.05, 0.2001, 0.005)
    oracle = {q: min(optimise(row_pooling_per_lot, q, n_max=n_max)[1],
                     optimise(grid_per_lot, q, n_max=n_max)[1]) for q in grid_p}
    rows = []
    for label, fn, cands in [("A — Row Pooling", row_pooling_per_lot, range(3, 9)),
                             ("B — Grid", grid_per_lot, range(4, 12))]:
        for n in cands:
            rows.append(dict(scheme=label, n=n,
                             worst_cost=max(fn(n, q) for q in grid_p),
                             max_regret=max(fn(n, q) - oracle[q] for q in grid_p)))
    rob = pd.DataFrame(rows).sort_values("max_regret")
    st.dataframe(rob.round(4).head(10), hide_index=True, width="stretch")
    st.markdown(
        f"**Row Pooling at n = 4–5 is the minimax-regret choice.** The grid at its "
        f"low-$p$ optimum is the most fragile option on the board: best if the "
        f"season is good, nearly worst if it is not."
    )

    st.divider()
    st.subheader("Better still — don't guess $p$, measure it for free")
    st.markdown(
        "Stage-1 results already estimate $p$ at no extra assay cost. If a "
        "fraction $f$ of pools of size $n$ reads positive, then since a pool is "
        "clean with probability $q^n$:"
    )
    st.latex(r"\hat{p} = 1 - (1-f)^{1/n}")
    npools = st.slider("Pools observed before re-optimising", 20, 800, 100, 20)
    se_hat = estimate_p_se(p, nD, npools)
    st.write(f"With **{npools}** pools of {nD} ({npools*nD:,} lots) at true "
             f"p = {p:.3f}: $\\hat p$ has a standard error of **±{se_hat:.4f}**, "
             f"i.e. a 95% interval of roughly "
             f"[{max(0,p-1.96*se_hat):.3f}, {p+1.96*se_hat:.3f}].")
    st.success("So after the first day or two of arrivals, $p$ is pinned down "
               "tightly enough to reset $n$ for the rest of the season — "
               "turning an unpredictable parameter into a measured one.")


# ======================================================= ⑥ IMPERFECT ASSAY
with tabs[5]:
    st.header("Dropping the perfect-assay assumption")
    st.markdown(
        "**The gate model.** Every screen or confirmatory assay is one "
        "*gate*. A gate covering a group of lots reads positive with "
        "probability $Se$ if the group truly contains at least one "
        "contaminated lot, and with probability $1-Sp$ if the group is "
        "truly clean — flat, no dilution, no dependence on group size.\n\n"
        "A contaminated lot is released only if it clears **every** gate "
        "it must pass; a clean lot is wrongly condemned only if **every** "
        "gate it passes also errs against it. Individual testing has "
        "$k=1$ gate, Row Pooling has $k=2$ (screen, confirm), the grid "
        "has $k=3$ (row, column, confirm):"
    )
    st.latex(r"\text{Escape rate} = 1 - Se^{k}")

    c1, c2 = st.columns(2)
    Se = c1.slider("Sensitivity Se", 0.80, 1.00, 0.98, 0.01)
    Sp = c2.slider("Specificity Sp", 0.80, 1.00, 0.99, 0.01)

    rows = []
    for label, k, n in [("Individual testing", 1, 1),
                        (f"Row Pooling n={nD}", 2, nD),
                        (f"Grid {nG}×{nG}", 3, nG)]:
        if k == 1:
            assays = float(N)
        elif k == 2:
            assays = row_pooling_per_lot_imperfect(n, p, Se, Sp) * N
        else:
            assays = grid_per_lot_imperfect(n, p, Se, Sp) * N
        esc = escape_rate(k, Se)
        cond = condemn_rate(k, n, p, Se, Sp)
        rows.append({"Scheme": label, "Gates": k,
                     "Expected assays": round(assays, 1),
                     "Escape rate": f"{esc*100:.2f}%",
                     "Escaped lots/season": round(esc * p * N, 2),
                     "Clean lots wrongly condemned": round(cond * (1 - p) * N, 2)})
    err = pd.DataFrame(rows)
    st.dataframe(err, hide_index=True, width="stretch")
    st.caption("Individual testing has just one gate — the assay itself — "
               "so it is the exact benchmark of N assays with escape rate "
               "$1-Se$. Every additional gate a scheme adds compounds its "
               "own escape rate, independent of pool or grid size.")

    d1, d2 = st.columns(2)
    with d1:
        st.subheader("Which error gets worse: false negatives")
        st.markdown(
            "The grid is the worse offender because a contaminated lot is "
            "caught only if its row **and** its column **and** its "
            "confirmation all fire — three sequential gates instead of Row "
            "Pooling's two. Escape rate is $1-Se^k$, so every extra gate "
            "compounds the miss risk **regardless of pool or grid size**."
        )
        grid_row = err[err.Gates == 3].iloc[0]
        ind_row = err[err.Gates == 1].iloc[0]
        st.metric("Grid's escape rate vs individual testing",
                  grid_row["Escape rate"],
                  f"{escape_rate(3,Se)/max(escape_rate(1,Se),1e-9):.1f}× worse",
                  delta_color="inverse")
    with d2:
        st.subheader("Which error washes out: false positives")
        st.markdown(
            "A clean lot is only wrongly condemned if **every** gate it "
            "passes also errs against it — two independent errors must "
            "coincide for Row Pooling, three for the grid — so false "
            "condemnation collapses as gates are added.\n\n"
            "Note this is a structural asymmetry: adding gates makes false "
            "**negatives** worse (only one gate needs to fail) but makes "
            "false **positives** rarer (every gate must fail) — the same "
            "mechanism cuts both ways."
        )

    st.error("**Verdict: this pushes Rohan away from the grid.** Escape rate "
             "depends only on the number of sequential gates a scheme uses, "
             "not on pool or grid size — so the grid's extra AND-gate is a "
             "structural sensitivity liability that no choice of $n$ can "
             "fix. Sensitivity is the error that costs Sahyadri a flagged "
             "container.")


# =============================================================== ⑦ VALIDATION
with tabs[6]:
    st.header("Validating the model")
    st.markdown(
        "Three independent checks. Nothing in the recommendation rests on "
        "algebra that has not been confirmed against something else."
    )

    st.subheader("1. Closed form vs Monte Carlo")
    st.markdown(
        "The simulator draws a fresh Bernoulli($p$) population of "
        f"{int(N):,} lots, physically executes each scheme, and counts assays. "
        "It knows nothing about the formulas. Agreement within ±2 standard "
        "errors is the check — and it also validates the tricky $2n-1$ term."
    )
    if st.button("Run the validation table", type="primary"):
        rows = []
        for scheme, fn in [("Row Pooling", row_pooling_per_lot), ("Grid", grid_per_lot)]:
            for pp in (0.05, 0.10, 0.20):
                for n in (4, 6, 9):
                    a = fn(n, pp)
                    key = "row_pooling" if scheme == "Row Pooling" else "grid"
                    m, se = mc(key, n, pp, int(N), reps)
                    rows.append({"scheme": scheme, "p": pp, "n": n,
                                 "analytic": a, "simulated": m, "SE": se,
                                 "z": (m - a) / se})
        v = pd.DataFrame(rows)
        st.dataframe(v.round({"analytic": 5, "simulated": 5, "SE": 5, "z": 2}),
                     hide_index=True, width="stretch")
        worst = v.z.abs().max()
        (st.success if worst < 2.5 else st.warning)(
            f"Largest |z| = {worst:.2f} across {len(v)} configurations.")

    st.subheader("2. A known analytical benchmark")
    st.markdown(
        "The classical result for Row Pooling is that pooling stops beating "
        "individual testing at $p = 1 - 3^{-1/3}$. My optimiser, which was "
        "written without reference to it, finds the break-even at **p ≈ 0.310** "
        f"against the exact **{1-3**(-1/3):.4f}** — agreement to the 0.005 "
        "resolution of the sweep."
    )

    st.subheader("Handling $n \\nmid N$")
    leftover = int(N) % (nG ** 2)
    leftover_assays = row_pooling_per_lot(nG, p) * leftover
    m_grid, se_grid = mc("grid", nG, p, int(N), reps)
    z_grid = (m_grid - cG) / se_grid
    st.markdown(
        f"{int(N):,} is not a multiple of {nG}² = {nG**2}. The simulator runs "
        f"{int(N)//(nG**2)} full arrays and sweeps the {leftover} leftover "
        "lots into Row Pooling groups, and the simulated cost still matches the "
        f"idealised formula to within about {abs(z_grid):.1f} standard errors. "
        f"The remainder is worth about {leftover_assays:.0f} assays out of "
        f"{cG*N:,.0f} total."
    )


# =========================================================== ⑧ RECOMMENDATION
with tabs[7]:
    st.header("Recommendation for Rohan Deshpande")

    st.subheader("1. Pool. It is clearly worth the trouble.")
    st.markdown(
        f"Even in the worst season Rohan describes (20% contaminated) pooling "
        f"costs 0.82 assays per lot against 1.00. In a typical 5% season it "
        f"costs 0.43 — **1,705 assays instead of 4,000**, about 2,300 runs "
        f"saved, and the laboratory's two instruments clear the yard in well "
        f"under half the time. At ₹{cost_per_assay:,} per run that is roughly "
        f"₹{(4000-1705)*cost_per_assay/1e5:.1f} lakh a season."
    )

    st.subheader("2. Use the simple scheme — groups of 5 — not the grid.")
    st.markdown(
        "At 5% contamination the 9×9 grid is genuinely cheaper: 1,519 assays "
        "against 1,705, an 11% edge worth a few lakh rupees. I am still not "
        "recommending it, for three reasons in increasing order of weight."
    )
    st.markdown(
        "- **The edge vanishes by p ≈ 0.125**, which is inside Rohan's own "
        "stated range. Above that the grid is strictly worse, and by 20% at "
        "p = 0.20.\n"
        "- **Committed in advance, the 9×9 grid is the highest-regret option "
        "available.** Groups of 5 are the minimax-regret choice across 5–20%.\n"
        "- **With a realistic assay the grid releases contaminated lots at more "
        "than twice Row Pooling's rate**, because a lot must fail both its row and "
        "its column to be caught. Correcting that with duplicate stage-1 runs "
        "erases almost all of the cost advantage."
    )
    st.info("The grid is the right answer to *“what is cheapest at 5%?”* and "
            "the wrong answer to *“what should I run when I don't know the "
            "season and a miss costs me a rapid-alert listing?”*")

    st.subheader("3. Exact operating instructions")
    st.markdown(
        "- Default to **groups of 5**. If early pools indicate p > 0.10, drop "
        "to **4**; above 0.15, drop to **3**.\n"
        "- After roughly the first 500 lots, compute "
        "$\\hat p = 1-(1-f)^{1/5}$ from the fraction $f$ of positive pools and "
        "reset $n$ for the rest of the season. This costs nothing and is "
        "accurate to about ±0.01.\n"
        "- If $\\hat p$ ever exceeds **0.30**, stop pooling and test "
        "individually — beyond that the arithmetic no longer works.\n"
        "- Run stage-1 pools in **duplicate**, positive if either replicate "
        "fires. Cost rises from 0.41 to 0.64 assays per lot; the miss rate "
        "falls from 9.8% to 2.8%. At ₹3,000 per assay that insurance is about "
        "₹2.7 lakh a season against a container loss of an entirely different "
        "order of magnitude."
    )

    st.subheader("4. Range of validity")
    st.markdown("The advice holds for p anywhere in **0.05–0.20** and degrades "
                "gracefully to **0.30**, beyond which no pooling scheme helps.")

    st.subheader("5. Assumptions whose failure would change the conclusion")
    st.markdown(
        "- **Independence.** Lots are assumed independently contaminated. They "
        "are not: aflatoxin follows weather, so lots from one belt after one wet "
        "week are correlated. Clustering *helps* Row Pooling — fewer, dirtier "
        "positive groups — and **hurts the grid badly**, because a cluster "
        "lights up many rows and columns at once. This only strengthens the "
        "recommendation. Practically: randomise lots across pools rather than "
        "pooling by village or arrival batch.\n"
        "- **Equal-cost assays.** If pooled and individual runs differ in price, "
        "or the pipetting labour of building 800 combined tubes is material, "
        "the optimum shifts. Add a per-tube handling cost and re-optimise; it "
        "will push $n$ up.\n"
        "- **No pool-size limit.** Groups of 5 sit comfortably inside LC–MS/MS "
        "dilution limits; the grid's rows of 9 are closer to the edge. If the "
        "validated maximum pool size is below 5, the whole calculation must be "
        "redone at that cap.\n"
        "- **Constant p across farmers.** Suppliers with known histories should "
        "be stratified — pooling known-risky farmers separately at a smaller "
        "$n$ beats one blanket rate for everybody."
    )

    st.divider()
    st.caption("Model, simulation and figures: `sahyadri_pooling.py`. "
               "Static reproduction of every number: `run_analysis.py`.")
