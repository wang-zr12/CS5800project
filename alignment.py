"""
alignment.py — Core DP implementations of Needleman-Wunsch, Smith-Waterman, and Gotoh algorithms.

Needleman-Wunsch (global):
  F[i,0] = i*gap,  F[0,j] = j*gap
  F[i,j] = max(F[i-1,j-1]+s, F[i-1,j]+gap, F[i,j-1]+gap)
  Traceback: (m,n) → (0,0)

Smith-Waterman (local):
  H[i,j] = max(0, H[i-1,j-1]+s, H[i-1,j]+gap, H[i,j-1]+gap)
  Traceback: argmax(H) → first 0 cell

Gotoh (affine gap penalties):
  Affine gap cost:  w(k) = gap_open + (k - 1) * gap_extend
  Three matrices: M (match/mismatch), Ix (gap in seq2), Iy (gap in seq1)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

NEG_INF = float('-inf')


@dataclass
class ScoringParams:
    match: int = 2
    mismatch: int = -1
    gap: int = -2


@dataclass
class AffineScoringParams:
    match: int = 2
    mismatch: int = -1
    gap_open: int = -10
    gap_extend: int = -1


@dataclass
class GeneralGapParams:
    """
    Alignment parameters for an arbitrary gap-penalty function.

    ``gap_fn(k)`` returns the penalty for a contiguous gap of length k (k ≥ 1).
    By convention the return value is ≤ 0.

    Four built-in shapes are available via class-methods:

    =========================================================
    Shape        Formula                  Factory
    ---------    -------------------------  -------------------
    Linear       w(k) = g · k             GeneralGapParams.linear()
    Affine       w(k) = g_o + (k-1)·g_e  GeneralGapParams.affine()
    Logarithmic  w(k) = c · log(k+1)      GeneralGapParams.logarithmic()
    Quadratic    w(k) = c · k²            GeneralGapParams.quadratic()
    =========================================================

    Complexity: O(m²n + mn²) time, O(mn) space (see README).
    """
    match: int = 2
    mismatch: int = -1
    gap_fn: object = None        # callable: int -> float
    gap_fn_name: str = 'general'

    def __post_init__(self):
        if self.gap_fn is None:
            self.gap_fn = lambda k: -2.0 * k   # default: linear w/ gap=-2

    # ── Factory methods ───────────────────────────────────────────────────

    @classmethod
    def linear(cls, gap: float = -2.0,
               match: int = 2, mismatch: int = -1) -> 'GeneralGapParams':
        """w(k) = gap × k  —  identical to NW/SW linear-gap recurrence."""
        _g = gap
        return cls(match=match, mismatch=mismatch,
                   gap_fn=lambda k: _g * k,
                   gap_fn_name=f'linear(g={gap})')

    @classmethod
    def affine(cls, gap_open: float = -10.0, gap_extend: float = -1.0,
               match: int = 2, mismatch: int = -1) -> 'GeneralGapParams':
        """w(k) = gap_open + (k-1) × gap_extend  —  same shape as Gotoh."""
        _o, _e = gap_open, gap_extend
        return cls(match=match, mismatch=mismatch,
                   gap_fn=lambda k: _o + (k - 1) * _e,
                   gap_fn_name=f'affine(o={gap_open},e={gap_extend})')

    @classmethod
    def logarithmic(cls, coeff: float = -3.0,
                    match: int = 2, mismatch: int = -1) -> 'GeneralGapParams':
        """w(k) = coeff × log(k+1)  —  concave; cheapest per-base for long gaps."""
        _c = coeff
        return cls(match=match, mismatch=mismatch,
                   gap_fn=lambda k: _c * math.log(k + 1),
                   gap_fn_name=f'log(c={coeff})')

    @classmethod
    def quadratic(cls, coeff: float = -0.05,
                  match: int = 2, mismatch: int = -1) -> 'GeneralGapParams':
        """w(k) = coeff × k²  —  convex; heavily penalises long gaps."""
        _c = coeff
        return cls(match=match, mismatch=mismatch,
                   gap_fn=lambda k: _c * k * k,
                   gap_fn_name=f'quadratic(c={coeff})')


@dataclass
class AlignmentResult:
    algorithm: str
    seq1: str
    seq2: str
    seq1_aligned: str
    seq2_aligned: str
    score: float
    matrix: np.ndarray
    traceback_path: List[Tuple[int, int]]
    params: ScoringParams


def _sub(a: str, b: str, p: ScoringParams) -> int:
    return p.match if a == b else p.mismatch


def _sub_affine(a: str, b: str, p: AffineScoringParams) -> int:
    return p.match if a == b else p.mismatch


def _build_display_matrix(M: np.ndarray, Ix: np.ndarray, Iy: np.ndarray,
                          clamp: float = -9999) -> np.ndarray:
    """Element-wise max of the three matrices, with -inf clamped."""
    safe_M = np.where(np.isinf(M), clamp, M)
    safe_Ix = np.where(np.isinf(Ix), clamp, Ix)
    safe_Iy = np.where(np.isinf(Iy), clamp, Iy)
    return np.maximum(safe_M, np.maximum(safe_Ix, safe_Iy)).astype(int)


# ── Needleman-Wunsch (Global Alignment) ─────────────────────────────────────

def needleman_wunsch(seq1: str, seq2: str, params: ScoringParams = None) -> AlignmentResult:
    """Global alignment via Needleman-Wunsch."""
    if params is None:
        params = ScoringParams()

    m, n = len(seq1), len(seq2)
    F = np.zeros((m + 1, n + 1), dtype=int)

    # Border initialisation: linear gap penalty
    for i in range(m + 1):
        F[i, 0] = i * params.gap
    for j in range(n + 1):
        F[0, j] = j * params.gap

    # Fill
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            diag = F[i - 1, j - 1] + _sub(seq1[i - 1], seq2[j - 1], params)
            up = F[i - 1, j] + params.gap
            left = F[i, j - 1] + params.gap
            F[i, j] = max(diag, up, left)

    # Traceback: (m,n) → (0,0), priority diagonal > up > left
    path: List[Tuple[int, int]] = []
    i, j = m, n
    s1: List[str] = []
    s2: List[str] = []

    while i > 0 or j > 0:
        path.append((i, j))
        if (i > 0 and j > 0
                and F[i, j] == F[i - 1, j - 1] + _sub(seq1[i - 1], seq2[j - 1], params)):
            s1.append(seq1[i - 1])
            s2.append(seq2[j - 1])
            i -= 1;
            j -= 1
        elif i > 0 and F[i, j] == F[i - 1, j] + params.gap:
            s1.append(seq1[i - 1])
            s2.append('-')
            i -= 1
        else:
            s1.append('-')
            s2.append(seq2[j - 1])
            j -= 1

    path.append((0, 0))
    path.reverse()
    s1.reverse()
    s2.reverse()

    return AlignmentResult(
        algorithm='Needleman-Wunsch',
        seq1=seq1, seq2=seq2,
        seq1_aligned=''.join(s1),
        seq2_aligned=''.join(s2),
        score=int(F[m, n]),
        matrix=F,
        traceback_path=path,
        params=params,
    )


# ── Smith-Waterman (Local Alignment) ────────────────────────────────────────

def smith_waterman(seq1: str, seq2: str, params: ScoringParams = None) -> AlignmentResult:
    """Local alignment via Smith-Waterman."""
    if params is None:
        params = ScoringParams()

    m, n = len(seq1), len(seq2)
    H = np.zeros((m + 1, n + 1), dtype=int)

    # Fill — key difference: floor at 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            diag = H[i - 1, j - 1] + _sub(seq1[i - 1], seq2[j - 1], params)
            up = H[i - 1, j] + params.gap
            left = H[i, j - 1] + params.gap
            H[i, j] = max(0, diag, up, left)

    max_val = int(H.max())

    # No local alignment possible (all zeros beyond border)
    if max_val == 0:
        return AlignmentResult(
            algorithm='Smith-Waterman',
            seq1=seq1, seq2=seq2,
            seq1_aligned='', seq2_aligned='',
            score=0,
            matrix=H,
            traceback_path=[],
            params=params,
        )

    # Traceback from highest-scoring cell until 0 is reached
    max_i, max_j = map(int, np.unravel_index(np.argmax(H), H.shape))
    i, j = max_i, max_j
    path: List[Tuple[int, int]] = []
    s1: List[str] = []
    s2: List[str] = []

    while H[i, j] > 0:
        path.append((i, j))
        if (i > 0 and j > 0
                and H[i, j] == H[i - 1, j - 1] + _sub(seq1[i - 1], seq2[j - 1], params)):
            s1.append(seq1[i - 1])
            s2.append(seq2[j - 1])
            i -= 1;
            j -= 1
        elif i > 0 and H[i, j] == H[i - 1, j] + params.gap:
            s1.append(seq1[i - 1])
            s2.append('-')
            i -= 1
        else:
            s1.append('-')
            s2.append(seq2[j - 1])
            j -= 1

    path.append((i, j))
    path.reverse()
    s1.reverse()
    s2.reverse()

    return AlignmentResult(
        algorithm='Smith-Waterman',
        seq1=seq1, seq2=seq2,
        seq1_aligned=''.join(s1),
        seq2_aligned=''.join(s2),
        score=max_val,
        matrix=H,
        traceback_path=path,
        params=params,
    )


# ── Gotoh Global (Affine Gap Penalties) ─────────────────────────────────────

def gotoh_global(seq1: str, seq2: str,
                 params: AffineScoringParams = None) -> AlignmentResult:
    """Global alignment with affine gap penalty (Gotoh's algorithm)."""
    if params is None:
        params = AffineScoringParams()

    go, ge = params.gap_open, params.gap_extend
    m, n = len(seq1), len(seq2)

    M = np.full((m + 1, n + 1), NEG_INF)
    Ix = np.full((m + 1, n + 1), NEG_INF)
    Iy = np.full((m + 1, n + 1), NEG_INF)

    # Border initialisation
    M[0, 0] = 0
    for i in range(1, m + 1):
        Ix[i, 0] = go + i * ge  # open one gap, extend i times
    for j in range(1, n + 1):
        Iy[0, j] = go + j * ge

    # Fill
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            s = _sub_affine(seq1[i - 1], seq2[j - 1], params)

            # Ix: gap in seq2 (consume seq1[i], skip seq2)
            Ix[i, j] = max(
                M[i - 1, j] + go + ge,
                Ix[i - 1, j] + ge,
            )

            # Iy: gap in seq1 (skip seq1, consume seq2[j])
            Iy[i, j] = max(
                M[i, j - 1] + go + ge,
                Iy[i, j - 1] + ge,
            )

            # M: match / mismatch
            M[i, j] = max(
                M[i - 1, j - 1],
                Ix[i - 1, j - 1],
                Iy[i - 1, j - 1],
            ) + s

    # Optimal score
    best = max(M[m, n], Ix[m, n], Iy[m, n])

    # Determine starting state
    if best == M[m, n]:
        state = 'M'
    elif best == Ix[m, n]:
        state = 'Ix'
    else:
        state = 'Iy'

    # Traceback
    path: List[Tuple[int, int]] = []
    s1: List[str] = []
    s2: List[str] = []
    i, j = m, n

    while i > 0 or j > 0:
        path.append((i, j))

        if state == 'M':
            s1.append(seq1[i - 1])
            s2.append(seq2[j - 1])
            s = _sub_affine(seq1[i - 1], seq2[j - 1], params)
            # Which predecessor?
            val = M[i, j] - s
            i -= 1;
            j -= 1
            if val == M[i, j]:
                state = 'M'
            elif val == Ix[i, j]:
                state = 'Ix'
            else:
                state = 'Iy'

        elif state == 'Ix':
            s1.append(seq1[i - 1])
            s2.append('-')
            val = Ix[i, j]
            i -= 1
            if val == M[i, j] + go + ge:
                state = 'M'
            else:
                state = 'Ix'

        else:  # Iy
            s1.append('-')
            s2.append(seq2[j - 1])
            val = Iy[i, j]
            j -= 1
            if val == M[i, j] + go + ge:
                state = 'M'
            else:
                state = 'Iy'

    path.append((0, 0))
    path.reverse();
    s1.reverse();
    s2.reverse()

    # Build the single display matrix for visualisation
    display = _build_display_matrix(M, Ix, Iy)

    # Compatible ScoringParams for downstream consumers
    compat = ScoringParams(match=params.match, mismatch=params.mismatch,
                           gap=params.gap_open)

    return AlignmentResult(
        algorithm='Gotoh-NW',
        seq1=seq1, seq2=seq2,
        seq1_aligned=''.join(s1),
        seq2_aligned=''.join(s2),
        score=int(best),
        matrix=display,
        traceback_path=path,
        params=compat,
    )


# ── Gotoh Local (Affine Gap Penalties) ──────────────────────────────────────

def gotoh_local(seq1: str, seq2: str,
                params: AffineScoringParams = None) -> AlignmentResult:
    """Local alignment with affine gap penalty (Gotoh's algorithm)."""
    if params is None:
        params = AffineScoringParams()

    go, ge = params.gap_open, params.gap_extend
    m, n = len(seq1), len(seq2)

    M = np.zeros((m + 1, n + 1))
    Ix = np.full((m + 1, n + 1), NEG_INF)
    Iy = np.full((m + 1, n + 1), NEG_INF)

    # Fill
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            s = _sub_affine(seq1[i - 1], seq2[j - 1], params)

            Ix[i, j] = max(
                M[i - 1, j] + go + ge,
                Ix[i - 1, j] + ge,
            )

            Iy[i, j] = max(
                M[i, j - 1] + go + ge,
                Iy[i, j - 1] + ge,
            )

            M[i, j] = max(
                0,
                M[i - 1, j - 1] + s,
                Ix[i - 1, j - 1] + s,
                Iy[i - 1, j - 1] + s,
            )

    # Best score is always in M (local alignment ends on match/mismatch)
    max_val = int(M.max())
    if max_val == 0:
        display = np.maximum(0, _build_display_matrix(M, Ix, Iy))
        compat = ScoringParams(match=params.match, mismatch=params.mismatch,
                               gap=params.gap_open)
        return AlignmentResult(
            algorithm='Gotoh-SW', seq1=seq1, seq2=seq2,
            seq1_aligned='', seq2_aligned='', score=0,
            matrix=display, traceback_path=[], params=compat,
        )

    max_i, max_j = map(int, np.unravel_index(np.argmax(M), M.shape))

    # Traceback from max cell until M == 0
    path: List[Tuple[int, int]] = []
    s1: List[str] = []
    s2: List[str] = []
    i, j = max_i, max_j
    state = 'M'

    while True:
        if state == 'M' and M[i, j] == 0:
            break

        path.append((i, j))

        if state == 'M':
            s = _sub_affine(seq1[i - 1], seq2[j - 1], params)
            s1.append(seq1[i - 1])
            s2.append(seq2[j - 1])
            val = M[i, j] - s
            i -= 1;
            j -= 1
            if val == M[i, j]:
                state = 'M'
            elif val == Ix[i, j]:
                state = 'Ix'
            else:
                state = 'Iy'

        elif state == 'Ix':
            s1.append(seq1[i - 1])
            s2.append('-')
            val = Ix[i, j]
            i -= 1
            if val == M[i, j] + go + ge:
                state = 'M'
            else:
                state = 'Ix'

        else:  # Iy
            s1.append('-')
            s2.append(seq2[j - 1])
            val = Iy[i, j]
            j -= 1
            if val == M[i, j] + go + ge:
                state = 'M'
            else:
                state = 'Iy'

    path.append((i, j))
    path.reverse();
    s1.reverse();
    s2.reverse()

    display = np.maximum(0, _build_display_matrix(M, Ix, Iy))
    compat = ScoringParams(match=params.match, mismatch=params.mismatch,
                           gap=params.gap_open)

    return AlignmentResult(
        algorithm='Gotoh-SW',
        seq1=seq1, seq2=seq2,
        seq1_aligned=''.join(s1),
        seq2_aligned=''.join(s2),
        score=max_val,
        matrix=display,
        traceback_path=path,
        params=compat,
    )


# ── Needleman-Wunsch: Arbitrary Gap Penalty ─────────────────────────────────

def needleman_wunsch_general(seq1: str, seq2: str,
                              params: GeneralGapParams = None) -> AlignmentResult:
    """
    Global alignment with an arbitrary gap-penalty function (general gap NW).

    Recurrence:
        F[i,0] = gap_fn(i),   F[0,j] = gap_fn(j)
        F[i,j] = max(
            F[i-1,j-1] + sub(i,j),
            max_{k=1..i}( F[i-k, j] + gap_fn(k) ),   # gap in seq2
            max_{k=1..j}( F[i, j-k] + gap_fn(k) )    # gap in seq1
        )

    Time: O(m²n + mn²)   Space: O(mn)
    (Use Gotoh for affine gaps at O(mn) cost.)
    """
    if params is None:
        params = GeneralGapParams()

    g = params.gap_fn
    m, n = len(seq1), len(seq2)
    F = np.zeros((m + 1, n + 1), dtype=float)

    # ── Border init: each border cell = cost of a single gap of that length ──
    for i in range(1, m + 1):
        F[i, 0] = g(i)
    for j in range(1, n + 1):
        F[0, j] = g(j)

    # ── Fill ──────────────────────────────────────────────────────────────────
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            s = params.match if seq1[i - 1] == seq2[j - 1] else params.mismatch
            diag      = F[i - 1, j - 1] + s
            up_best   = max(F[i - k, j] + g(k) for k in range(1, i + 1))
            left_best = max(F[i, j - k] + g(k) for k in range(1, j + 1))
            F[i, j]   = max(diag, up_best, left_best)

    # ── Traceback: (m,n) → (0,0) ─────────────────────────────────────────────
    path: List[Tuple[int, int]] = []
    s1_aln: List[str] = []
    s2_aln: List[str] = []
    i, j = m, n

    while i > 0 or j > 0:
        path.append((i, j))
        if j == 0:
            # Entire remaining seq1 is a single border gap
            s1_aln.extend(reversed(seq1[:i]))
            s2_aln.extend(['-'] * i)
            i = 0
        elif i == 0:
            s1_aln.extend(['-'] * j)
            s2_aln.extend(reversed(seq2[:j]))
            j = 0
        else:
            s = params.match if seq1[i - 1] == seq2[j - 1] else params.mismatch
            if abs(F[i, j] - (F[i - 1, j - 1] + s)) < 1e-9:
                s1_aln.append(seq1[i - 1])
                s2_aln.append(seq2[j - 1])
                i -= 1; j -= 1
            else:
                # Try vertical gap of length k (gap in seq2)
                moved = False
                for k in range(1, i + 1):
                    if abs(F[i, j] - (F[i - k, j] + g(k))) < 1e-9:
                        s1_aln.extend(reversed(seq1[i - k:i]))
                        s2_aln.extend(['-'] * k)
                        i -= k
                        moved = True
                        break
                if not moved:
                    # Horizontal gap of length k (gap in seq1)
                    for k in range(1, j + 1):
                        if abs(F[i, j] - (F[i, j - k] + g(k))) < 1e-9:
                            s1_aln.extend(['-'] * k)
                            s2_aln.extend(reversed(seq2[j - k:j]))
                            j -= k
                            break

    path.append((0, 0))
    path.reverse();  s1_aln.reverse();  s2_aln.reverse()

    compat = ScoringParams(match=params.match, mismatch=params.mismatch,
                           gap=int(round(g(1))))
    return AlignmentResult(
        algorithm='NW-General',
        seq1=seq1, seq2=seq2,
        seq1_aligned=''.join(s1_aln),
        seq2_aligned=''.join(s2_aln),
        score=float(F[m, n]),
        matrix=F.astype(int),
        traceback_path=path,
        params=compat,
    )


# ── Smith-Waterman: Arbitrary Gap Penalty ────────────────────────────────────

def smith_waterman_general(seq1: str, seq2: str,
                            params: GeneralGapParams = None) -> AlignmentResult:
    """
    Local alignment with an arbitrary gap-penalty function (general gap SW).

    Recurrence:
        H[i,j] = max(
            0,
            H[i-1,j-1] + sub(i,j),
            max_{k=1..i}( H[i-k, j] + gap_fn(k) ),
            max_{k=1..j}( H[i, j-k] + gap_fn(k) )
        )
    Traceback: argmax(H) → first cell where H ≤ 0.

    Time: O(m²n + mn²)   Space: O(mn)
    """
    if params is None:
        params = GeneralGapParams()

    g = params.gap_fn
    m, n = len(seq1), len(seq2)
    H = np.zeros((m + 1, n + 1), dtype=float)

    # ── Fill ──────────────────────────────────────────────────────────────────
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            s = params.match if seq1[i - 1] == seq2[j - 1] else params.mismatch
            diag      = H[i - 1, j - 1] + s
            up_best   = max(H[i - k, j] + g(k) for k in range(1, i + 1))
            left_best = max(H[i, j - k] + g(k) for k in range(1, j + 1))
            H[i, j]   = max(0.0, diag, up_best, left_best)

    max_val = float(H.max())
    compat = ScoringParams(match=params.match, mismatch=params.mismatch,
                           gap=int(round(g(1))))

    if max_val == 0.0:
        return AlignmentResult(
            algorithm='SW-General', seq1=seq1, seq2=seq2,
            seq1_aligned='', seq2_aligned='', score=0.0,
            matrix=H.astype(int), traceback_path=[], params=compat,
        )

    max_i, max_j = map(int, np.unravel_index(np.argmax(H), H.shape))
    i, j = max_i, max_j
    path: List[Tuple[int, int]] = []
    s1_aln: List[str] = []
    s2_aln: List[str] = []

    # ── Traceback: argmax → first zero cell ───────────────────────────────────
    while H[i, j] > 0:
        path.append((i, j))
        s = params.match if seq1[i - 1] == seq2[j - 1] else params.mismatch
        if i > 0 and j > 0 and abs(H[i, j] - (H[i - 1, j - 1] + s)) < 1e-9:
            s1_aln.append(seq1[i - 1])
            s2_aln.append(seq2[j - 1])
            i -= 1; j -= 1
        else:
            moved = False
            # Try vertical gap (only if combined value is positive, matching H)
            for k in range(1, i + 1):
                candidate = H[i - k, j] + g(k)
                if candidate < 0.0:
                    continue
                if abs(H[i, j] - candidate) < 1e-9:
                    s1_aln.extend(reversed(seq1[i - k:i]))
                    s2_aln.extend(['-'] * k)
                    i -= k
                    moved = True
                    break
            if not moved:
                for k in range(1, j + 1):
                    candidate = H[i, j - k] + g(k)
                    if candidate < 0.0:
                        continue
                    if abs(H[i, j] - candidate) < 1e-9:
                        s1_aln.extend(['-'] * k)
                        s2_aln.extend(reversed(seq2[j - k:j]))
                        j -= k
                        break

    path.append((i, j))
    path.reverse();  s1_aln.reverse();  s2_aln.reverse()

    return AlignmentResult(
        algorithm='SW-General',
        seq1=seq1, seq2=seq2,
        seq1_aligned=''.join(s1_aln),
        seq2_aligned=''.join(s2_aln),
        score=max_val,
        matrix=H.astype(int),
        traceback_path=path,
        params=compat,
    )


# ── Utility Functions ───────────────────────────────────────────────────────

def calculate_alignment_stats(seq1_aligned: str, seq2_aligned: str) -> dict:
    matches = 0
    mismatches = 0
    gaps = 0

    alignment_length = len(seq1_aligned)

    for i in range(alignment_length):
        if seq1_aligned[i] == '-' or seq2_aligned[i] == '-':
            gaps += 1
        elif seq1_aligned[i] == seq2_aligned[i]:
            matches += 1
        else:
            mismatches += 1

    identity_percent = (matches / alignment_length * 100) if alignment_length > 0 else 0

    # Coverage 可以定义为非 Gap 字符占原始序列长度的比例
    valid_positions = alignment_length - gaps
    coverage_percent = (valid_positions / alignment_length * 100) if alignment_length > 0 else 0

    return {
        'matches': matches,
        'mismatches': mismatches,
        'gaps': gaps,
        'identity_percent': identity_percent,
        'coverage_percent': coverage_percent,
        'alignment_length': alignment_length,
        'valid_positions': valid_positions
    }

def format_alignment_display(result: AlignmentResult, line_length: int = 60) -> str:
    """Format alignment result for display."""
    lines = []
    seq1_aligned = result.seq1_aligned
    seq2_aligned = result.seq2_aligned

    for i in range(0, len(seq1_aligned), line_length):
        end_pos = min(i + line_length, len(seq1_aligned))

        # Extract segments
        seg1 = seq1_aligned[i:end_pos]
        seg2 = seq2_aligned[i:end_pos]

        # Create match line
        match_line = ""
        for j in range(len(seg1)):
            if seg1[j] == '-' or seg2[j] == '-':
                match_line += " "
            elif seg1[j] == seg2[j]:
                match_line += "|"
            else:
                match_line += "*"

        # Add to lines
        lines.append(f"Seq1: {seg1}")
        lines.append(f"      {match_line}")
        lines.append(f"Seq2: {seg2}")
        lines.append("")

    return "\n".join(lines)


# ── Algorithm Registry ──────────────────────────────────────────────────────

ALIGNMENT_ALGORITHMS = {
    'needleman_wunsch':          needleman_wunsch,
    'smith_waterman':            smith_waterman,
    'gotoh_global':              gotoh_global,
    'gotoh_local':               gotoh_local,
    'needleman_wunsch_general':  needleman_wunsch_general,
    'smith_waterman_general':    smith_waterman_general,
}


def get_algorithm(name: str):
    """Get alignment algorithm by name."""
    if name not in ALIGNMENT_ALGORITHMS:
        available = ', '.join(ALIGNMENT_ALGORITHMS.keys())
        raise ValueError(f"Unknown algorithm '{name}'. Available: {available}")
    return ALIGNMENT_ALGORITHMS[name]


def list_algorithms() -> List[str]:
    """List all available alignment algorithms."""
    return list(ALIGNMENT_ALGORITHMS.keys())