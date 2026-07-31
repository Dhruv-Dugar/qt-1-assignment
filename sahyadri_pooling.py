"""
Sahyadri Spice Company - Pooled Testing Model
=============================================

Notation follows the QT class notes:
    p : probability an individual lot is contaminated (i.i.d. across lots)
    q = 1 - p
    n : pool size (row pooling) or grid side (cross pooling)
    N : total number of lots (4000 for Sahyadri)

Cost unit = 1 assay = 1 LC-MS/MS run on one tube.

Scheme A (Row Pooling), notes p.1:
    E[tests per group of n] = 1 + n * (1 - q^n)
    E[tests total]          = N/n + N * (1 - q^n)

Scheme B (cross / grid pooling), notes pp.2-5:
    Stage 1 : 2n assays per n x n array
    Stage 2 : lot (i,j) retested iff row i AND col j both positive
    K = P(row i +ve AND col j +ve)
      = P(A) + P(B) - P(A u B)                       [inclusion-exclusion]
      = 2(1 - q^n) - (1 - q^(2n-1))                  [row u col = 2n-1 lots]
      = 1 - 2 q^n + q^(2n-1)
    E[tests per array] = 2n + n^2 K
    E[tests total]     = (N/n^2)(2n + n^2 K) = 2N/n + N K
"""

import numpy as np

# ----------------------------------------------------------------------
# 1. ANALYTIC EXPECTED COST (per lot; multiply by N for the population)
# ----------------------------------------------------------------------

def row_pooling_per_lot(n, p):
    """Scheme A. E[assays]/lot = 1/n + (1 - q^n)."""
    q = 1.0 - p
    return 1.0 / n + (1.0 - q ** n)


def grid_K(n, p):
    """P(lot (i,j) goes to stage 2) = 1 - 2q^n + q^(2n-1)."""
    q = 1.0 - p
    return 1.0 - 2.0 * q ** n + q ** (2 * n - 1)


def grid_per_lot(n, p):
    """Scheme B, exactly as derived in the notes. E[assays]/lot = 2/n + K."""
    return 2.0 / n + grid_K(n, p)


def grid_per_lot_deduced(n, p):
    """
    Scheme B with one free logical deduction:
    if EXACTLY one row and EXACTLY one column are positive, the culprit is
    forced to be the intersection lot -> its confirmatory assay is redundant.
    That event occurs iff exactly one lot in the array is contaminated:
        P = n^2 * p * q^(n^2 - 1)
    E[assays]/lot = 2/n + K - p*q^(n^2-1)
    Reported as a refinement; the headline recommendation uses grid_per_lot.
    """
    q = 1.0 - p
    return grid_per_lot(n, p) - p * q ** (n * n - 1)


# ----------------------------------------------------------------------
# 2. OPTIMISATION over pool size
# ----------------------------------------------------------------------

def optimise(cost_fn, p, n_min=2, n_max=60):
    """Grid search over integer n. Returns (n*, cost*). n is small and the
    objective is cheap + not guaranteed unimodal at all p, so exhaustive
    search is safer than calculus or a continuous optimiser."""
    ns = np.arange(n_min, n_max + 1)
    costs = np.array([cost_fn(int(n), p) for n in ns])
    k = int(np.argmin(costs))
    return int(ns[k]), float(costs[k])


def cost_curve(cost_fn, p, n_min=2, n_max=40):
    ns = np.arange(n_min, n_max + 1)
    return ns, np.array([cost_fn(int(n), p) for n in ns])


# ----------------------------------------------------------------------
# 3. MONTE CARLO SIMULATION (validates the algebra + handles n not | N)
# ----------------------------------------------------------------------

def _row_pooling_tests(status, n):
    """status: 1-D bool array of true contamination. Returns assay count.
    Leftover lots (N mod n) form one short final group of size r < n,
    handled by the same rule."""
    N = status.size
    full = (N // n) * n # nearest number 
    tests = 0
    if full:
        blocks = status[:full].reshape(-1, n)
        pos = blocks.any(axis=1)
        tests += blocks.shape[0] + n * pos.sum()
    r = N - full
    if r:
        tail = status[full:]
        tests += 1 + (r if tail.any() else 0)
    return int(tests)


def _grid_tests(status, n, deduce=False):
    """n x n cross pooling. Leftover lots (N mod n^2) are handled by row
    pooling at the same n -- a defensible fallback because a partial array
    cannot be row/column pooled."""
    N = status.size
    per = n * n
    full = (N // per) * per
    tests = 0
    if full:
        arrays = status[:full].reshape(-1, n, n)
        rows = arrays.any(axis=2)          # (g, n) positive rows
        cols = arrays.any(axis=1)          # (g, n) positive cols
        R = rows.sum(axis=1)
        C = cols.sum(axis=1)
        stage2 = R * C
        if deduce:
            stage2 = np.where((R == 1) & (C == 1), 0, stage2)
        tests += arrays.shape[0] * 2 * n + int(stage2.sum())
    if N - full:
        tests += _row_pooling_tests(status[full:], n)
    return int(tests)


def simulate(scheme, n, p, N=4000, reps=2000, seed=0, deduce=False):
    """Returns (mean assays per lot, standard error). Callers should pass an
    explicit seed for reproducibility."""
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for r in range(reps):
        status = rng.random(N) < p
        if scheme == "row_pooling":
            out[r] = _row_pooling_tests(status, n)
        elif scheme == "grid":
            out[r] = _grid_tests(status, n, deduce=deduce)
        else:
            raise ValueError(scheme)
    out /= N
    return out.mean(), out.std(ddof=1) / np.sqrt(reps)


# ----------------------------------------------------------------------
# 4. IMPERFECT ASSAY (gate model, no dilution)
# ----------------------------------------------------------------------
# Every screen or confirmatory assay is one "gate." A gate covering a group
# of m lots reads positive w.p. Se if the group truly contains at least one
# contaminated lot, and w.p. (1 - Sp) if the group is truly clean -- flat,
# size-independent, no per-tube dilution term.
#
# A contaminated lot is released only if it clears every gate it must pass;
# a clean lot is wrongly condemned only if every gate it passes also errs
# against it. k = number of sequential gates: individual testing k=1,
# row pooling k=2 (screen, confirm), grid k=3 (row, column, confirm).

def _gate_trigger_prob(m, p, Se, Sp):
    """P(a gate covering m lots reads positive)."""
    q = 1.0 - p
    dirty = 1.0 - q ** m
    return dirty * Se + (1.0 - dirty) * (1.0 - Sp)


def row_pooling_per_lot_imperfect(n, p, Se, Sp):
    """Scheme A, imperfect assay. E[assays]/lot = 1/n + P(pool triggers)."""
    return 1.0 / n + _gate_trigger_prob(n, p, Se, Sp)


def grid_per_lot_imperfect(n, p, Se, Sp):
    """Scheme B, imperfect assay. E[assays]/lot = 2/n + P(row)*P(col)."""
    r = _gate_trigger_prob(n, p, Se, Sp)
    return 2.0 / n + r ** 2


def escape_rate(k, Se):
    """P(a contaminated lot is released) = 1 - Se^k."""
    return 1.0 - Se ** k


def condemn_rate(k, n, p, Se, Sp):
    """P(a clean lot is wrongly declared contaminated)."""
    if k == 1:
        return 1.0 - Sp
    trigger = _gate_trigger_prob(n - 1, p, Se, Sp)   # conditioned on this lot being clean
    if k == 2:
        return trigger * (1.0 - Sp)
    if k == 3:
        return trigger ** 2 * (1.0 - Sp)
    raise ValueError(k)


# ----------------------------------------------------------------------
# 5. ESTIMATING p FROM THE POOL RESULTS THEMSELVES
# ----------------------------------------------------------------------
# A pool of n is negative w.p. q^n, so if a fraction f of pools reads
# positive, the MLE of p is  p_hat = 1 - (1 - f)^(1/n).
# This costs nothing extra: it reuses stage-1 results Sahyadri already pays for.

def estimate_p_se(p, n, n_pools):
    """Delta-method standard error of p_hat."""
    q = 1 - p
    f = 1 - q ** n
    dp_df = (1 - f) ** (1 / n - 1) / n
    return float(np.sqrt(f * (1 - f) / n_pools) * dp_df)
