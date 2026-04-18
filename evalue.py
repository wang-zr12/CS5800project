"""
evalue.py — Karlin-Altschul statistical significance for local alignments.

Core formula:   E = K · m · n · exp(-λ · S)
    E  = expected number of random alignments scoring >= S
    K  = database-size constant (estimated empirically)
    λ  = scoring-scheme normalisation constant (computed analytically)
    S  = raw alignment score
    m, n = sequence lengths

λ is computed by bisection: find the positive root of
    Σ_a Σ_b  p_a · p_b · exp(λ · s(a,b))  =  1

K is estimated by Monte-Carlo + Gumbel method-of-moments fit.

Reference: Karlin & Altschul (1990), PNAS 87:2264-2268.

NOTE: The theory is exact for ungapped local alignments.  For gapped
alignments (including affine gap), the same formula is widely used as
an approximation (this is what BLAST does), but the parameters are
empirical rather than analytical.
"""

import math
import random
from typing import Optional

import numpy as np

from alignment import ScoringParams, smith_waterman


# ── λ estimation (analytical, bisection) ─────────────────────────────────────

_BASES = ['A', 'C', 'G', 'T']


def compute_lambda(params: ScoringParams,
                   base_freqs: Optional[dict] = None,
                   tol: float = 1e-10,
                   max_iter: int = 200) -> float:
    """
    Solve  Σ_{a,b} p_a · p_b · exp(λ · s(a,b)) = 1  for the positive root λ.

    For uniform DNA frequencies (0.25 each) and match / mismatch scoring
    this simplifies to:
        0.25 · exp(λ·match) + 0.75 · exp(λ·mismatch) = 1

    Uses bisection on the interval (ε, upper].
    """
    if base_freqs is None:
        base_freqs = {b: 0.25 for b in _BASES}

    def f(lam: float) -> float:
        total = 0.0
        for a in _BASES:
            for b in _BASES:
                s_ab = params.match if a == b else params.mismatch
                total += base_freqs[a] * base_freqs[b] * math.exp(lam * s_ab)
        return total - 1.0

    # f(0) = 0 (trivial root), f'(0) < 0 when E[score] < 0 → f dips below 0
    # We need the second, positive root.
    lo = 1e-6
    hi = 2.0
    while f(hi) <= 0 and hi < 1e6:
        hi *= 2.0

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break

    return (lo + hi) / 2.0


# ── K estimation (Monte-Carlo + Gumbel fit) ──────────────────────────────────

_EULER_MASCHERONI = 0.5772156649015329


def estimate_K(params: ScoringParams,
               lam: float,
               seq_length: int = 100,
               num_pairs: int = 1000,
               seed: int = 42) -> float:
    """
    Estimate K by running Smith-Waterman on many random sequence pairs
    and fitting the resulting scores to a Gumbel distribution via the
    method of moments.

    Gumbel CDF:  F(x) = exp( -exp( -(x - μ)/β ) )
        mean     = μ + β · γ        (γ = Euler-Mascheroni)
        variance = (π² / 6) · β²

    From Karlin-Altschul:  μ = ln(K · m · n) / λ ,  β = 1 / λ
    ⇒  K = exp(μ · λ) / (m · n)
    """
    rng = random.Random(seed)
    scores = []

    for _ in range(num_pairs):
        s1 = ''.join(rng.choices(_BASES, k=seq_length))
        s2 = ''.join(rng.choices(_BASES, k=seq_length))
        result = smith_waterman(s1, s2, params)
        scores.append(float(result.score))

    arr = np.array(scores, dtype=float)
    sample_mean = float(np.mean(arr))
    sample_var  = float(np.var(arr, ddof=1))

    # Fit Gumbel
    beta = math.sqrt(6.0 * max(sample_var, 1e-12)) / math.pi
    mu   = sample_mean - beta * _EULER_MASCHERONI

    K = math.exp(mu * lam) / (seq_length * seq_length)
    return max(K, 1e-20)       # clamp to avoid zero


# ── E-value and bit score ────────────────────────────────────────────────────

def compute_evalue(score: float, m: int, n: int,
                   lam: float, K: float) -> float:
    """E = K · m · n · exp(-λ · S)"""
    return K * m * n * math.exp(-lam * score)


def compute_bit_score(score: float, lam: float, K: float) -> float:
    """bits = (λ · S  -  ln K) / ln 2"""
    return (lam * score - math.log(K)) / math.log(2)


def interpret_evalue(evalue: float) -> str:
    """Human-readable significance label."""
    if evalue < 1e-50:
        return 'Extremely significant'
    if evalue < 1e-10:
        return 'Highly significant'
    if evalue < 0.01:
        return 'Significant'
    if evalue < 1.0:
        return 'Possibly significant'
    if evalue < 10.0:
        return 'Not significant'
    return 'No meaningful similarity'


def evaluate_alignment(result, lam: float, K: float) -> dict:
    """
    Compute E-value, bit score and interpretation for an AlignmentResult.

    Returns a dict with keys:
        score, seq1_len, seq2_len, lambda, K, evalue, bits, interpretation
    """
    m = len(result.seq1)
    n = len(result.seq2)
    ev   = compute_evalue(result.score, m, n, lam, K)
    bits = compute_bit_score(result.score, lam, K)

    return {
        'score':          result.score,
        'seq1_len':       m,
        'seq2_len':       n,
        'lambda':         lam,
        'K':              K,
        'evalue':         ev,
        'bits':           round(bits, 1),
        'interpretation': interpret_evalue(ev),
    }
