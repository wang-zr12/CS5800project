"""
visualization.py — Clean, overlay-free plotting functions.

Each figure contains ONLY the DP heatmap, cell values, traceback path,
axis labels and a colorbar.  No text boxes, no legends, no descriptions.
Colour meaning is documented in the README, not on the figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from typing import List

from alignment import AlignmentResult


# ── Text helpers ─────────────────────────────────────────────────────────────

def _midline(s1: str, s2: str) -> str:
    out = []
    for a, b in zip(s1, s2):
        if a == '-' or b == '-':
            out.append(' ')
        elif a == b:
            out.append('|')
        else:
            out.append('.')
    return ''.join(out)


def format_alignment(result: AlignmentResult) -> str:
    s1  = result.seq1_aligned
    s2  = result.seq2_aligned
    mid = _midline(s1, s2)

    length     = len(s1)
    matches    = mid.count('|')
    mismatches = mid.count('.')
    gaps       = s1.count('-') + s2.count('-')
    identity   = matches / length * 100 if length > 0 else 0.0

    return (
        f"Algorithm  : {result.algorithm}\n"
        f"Score      : {result.score}\n"
        f"Length     : {length}\n"
        f"Identity   : {matches}/{length} ({identity:.1f}%)\n"
        f"Mismatches : {mismatches}\n"
        f"Gaps       : {gaps}\n"
        f"\n"
        f"Seq1: {s1}\n"
        f"      {mid}\n"
        f"Seq2: {s2}"
    )


def compute_metrics(result: AlignmentResult) -> dict:
    """Return a dict of alignment quality metrics."""
    s1  = result.seq1_aligned
    s2  = result.seq2_aligned
    mid = _midline(s1, s2)

    length     = len(s1) or 1
    matches    = mid.count('|')
    mismatches = mid.count('.')
    gaps       = s1.count('-') + s2.count('-')
    identity   = matches / length * 100
    gap_rate   = gaps / length * 100
    score_den  = result.score / length
    coverage   = length / max(len(result.seq1), len(result.seq2)) * 100

    return {
        'score':         result.score,
        'aligned_len':   length,
        'matches':       matches,
        'mismatches':    mismatches,
        'gaps':          gaps,
        'identity_pct':  round(identity, 1),
        'gap_rate_pct':  round(gap_rate, 1),
        'score_density': round(score_den, 2),
        'coverage_pct':  round(coverage, 1),
    }


# ── Colormaps ────────────────────────────────────────────────────────────────

_CMAPS = {
    'Needleman-Wunsch': 'YlOrRd',
    'Smith-Waterman':   'Blues',
    'Gotoh-NW':         'YlOrRd',
    'Gotoh-SW':         'Blues',
    'NW-General':       'YlOrRd',
    'SW-General':       'Blues',
}


# ── Single-matrix figure (clean, no overlays) ───────────────────────────────

def plot_single_matrix(result: AlignmentResult, save_path: str,
                       figsize: tuple = (9, 7.5)) -> None:
    """
    One clean DP-matrix heatmap per figure.

    Contents: heatmap · cell values · traceback path · axis labels · colorbar.
    No legends, no text boxes, no descriptions.
    """
    matrix = result.matrix
    seq1, seq2 = result.seq1, result.seq2
    m, n = len(seq1), len(seq2)
    cmap = _CMAPS.get(result.algorithm, 'viridis')

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix, cmap=cmap, aspect='auto', origin='upper')
    cbar = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    # ── Cell values (small matrices only) ────────────────────────────────
    if (m + 1) * (n + 1) <= 400:
        vmin, vmax = float(matrix.min()), float(matrix.max())
        span = (vmax - vmin) if vmax != vmin else 1.0
        for i in range(m + 1):
            for j in range(n + 1):
                norm = (matrix[i, j] - vmin) / span
                colour = 'white' if norm > 0.55 else 'black'
                ax.text(
                    j, i, str(int(matrix[i, j])),
                    ha='center', va='center',
                    fontsize=7, fontfamily='monospace',
                    color=colour, fontweight='bold',
                )

    # ── Traceback path ───────────────────────────────────────────────────
    path = result.traceback_path
    if len(path) > 1:
        rows = [p[0] for p in path]
        cols = [p[1] for p in path]

        # Outline for contrast
        ax.plot(cols, rows, '-', color='white', linewidth=3.5,
                alpha=0.55, zorder=4)
        # Main path
        ax.plot(cols, rows, '-', color='cyan', linewidth=2.0,
                alpha=0.92, zorder=5)

        # Start / end markers
        ax.plot(cols[0], rows[0], 'o', color='#2ca02c',
                markersize=8, zorder=6,
                path_effects=[pe.withStroke(linewidth=1.8, foreground='white')])
        ax.plot(cols[-1], rows[-1], 's', color='#d62728',
                markersize=8, zorder=6,
                path_effects=[pe.withStroke(linewidth=1.8, foreground='white')])

    # ── Axis labels ──────────────────────────────────────────────────────
    ax.set_xticks(range(n + 1))
    ax.set_xticklabels(['-'] + list(seq2), fontsize=9, fontfamily='monospace')
    ax.set_yticks(range(m + 1))
    ax.set_yticklabels(['-'] + list(seq1), fontsize=9, fontfamily='monospace')
    ax.set_xlabel('Sequence 2', fontsize=10)
    ax.set_ylabel('Sequence 1', fontsize=10)
    ax.set_title(result.algorithm, fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {save_path}')


# ── Score-comparison bar chart ───────────────────────────────────────────────

def plot_score_comparison(data: List[dict], save_path: str = None) -> None:
    """Grouped bar chart: NW vs SW score per experiment."""
    names     = [d['name'] for d in data]
    nw_scores = [d['nw_score'] for d in data]
    sw_scores = [d['sw_score'] for d in data]

    x     = np.arange(len(names))
    width = 0.32

    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x - width / 2, nw_scores, width,
                label='Needleman-Wunsch', color='#d62728',
                alpha=0.85, edgecolor='white', linewidth=0.8)
    b2 = ax.bar(x + width / 2, sw_scores, width,
                label='Smith-Waterman', color='#1f77b4',
                alpha=0.85, edgecolor='white', linewidth=0.8)

    ax.bar_label(b1, fmt='%d', fontsize=9, padding=3)
    ax.bar_label(b2, fmt='%d', fontsize=9, padding=3)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_xlabel('Experiment', fontsize=11)
    ax.set_ylabel('Alignment Score', fontsize=11)
    ax.set_title('Score Comparison: NW vs SW', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved -> {save_path}')


# ── Timing benchmark plot ────────────────────────────────────────────────────

def plot_timing(timing_data: List[tuple], save_path: str = None) -> None:
    """
    Line chart: execution time vs sequence length for NW and SW.
    timing_data: list of (length, nw_seconds, sw_seconds).
    """
    lengths   = [t[0] for t in timing_data]
    nw_times  = [t[1] * 1000 for t in timing_data]   # ms
    sw_times  = [t[2] * 1000 for t in timing_data]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(lengths, nw_times, 'o-', color='#d62728', linewidth=2,
            markersize=6, label='Needleman-Wunsch')
    ax.plot(lengths, sw_times, 's--', color='#1f77b4', linewidth=2,
            markersize=6, label='Smith-Waterman')

    ax.set_xlabel('Sequence length  n (= m)', fontsize=11)
    ax.set_ylabel('Time (ms)', fontsize=11)
    ax.set_title('Execution Time vs Sequence Length', fontsize=12,
                 fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved -> {save_path}')


# ── Multi-method timing benchmark plot ───────────────────────────────────────

_TIMING_STYLES = [
    {'color': '#d62728', 'marker': 'o',  'ls': '-'},    # NW linear
    {'color': '#1f77b4', 'marker': 's',  'ls': '-'},    # SW linear
    {'color': '#ff7f0e', 'marker': '^',  'ls': '--'},   # Gotoh-NW
    {'color': '#2ca02c', 'marker': 'D',  'ls': '--'},   # Gotoh-SW
    {'color': '#9467bd', 'marker': 'v',  'ls': '-.'},   # Biopython
    {'color': '#8c564b', 'marker': 'P',  'ls': ':'},    # General-NW  O(n³)
    {'color': '#e377c2', 'marker': 'X',  'ls': ':'},    # General-SW  O(n³)
]


def plot_timing_multi(timing_data: dict, save_path: str = None) -> None:
    """
    Multi-method timing comparison with log-scale y-axis.

    Parameters
    ----------
    timing_data : dict
        {method_name: [(length, seconds), ...], ...}
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    for idx, (name, points) in enumerate(timing_data.items()):
        style = _TIMING_STYLES[idx % len(_TIMING_STYLES)]
        lengths = [p[0] for p in points]
        times   = [p[1] * 1000 for p in points]    # seconds -> ms
        ax.plot(lengths, times,
                marker=style['marker'], linestyle=style['ls'],
                color=style['color'], linewidth=2, markersize=5,
                label=name, alpha=0.9)

    ax.set_yscale('log')
    ax.set_xlabel('Sequence length  n (= m)', fontsize=11)
    ax.set_ylabel('Time (ms, log scale)', fontsize=11)
    ax.set_title('Multi-Method Timing Comparison', fontsize=12,
                 fontweight='bold')
    ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5),
              borderaxespad=0, framealpha=0.9)
    ax.grid(True, which='both', alpha=0.25)
    ax.set_axisbelow(True)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved -> {save_path}')


# ── Large-scale validation study plot ────────────────────────────────────────

def plot_validation_study(data: List[dict], save_path: str = None) -> None:
    """
    Three-panel figure summarising the controlled-similarity validation.

    Parameters
    ----------
    data : list of dicts, one per similarity level, each with keys:
        similarity, nw_score_mean, sw_score_mean,
        nw_identity_mean, sw_identity_mean,
        nw_gap_rate_mean, sw_gap_rate_mean
    """
    sims      = [d['similarity'] * 100 for d in data]
    nw_scores = [d['nw_score_mean'] for d in data]
    sw_scores = [d['sw_score_mean'] for d in data]
    nw_ids    = [d['nw_identity_mean'] for d in data]
    sw_ids    = [d['sw_identity_mean'] for d in data]
    nw_gaps   = [d['nw_gap_rate_mean'] for d in data]
    sw_gaps   = [d['sw_gap_rate_mean'] for d in data]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Large-Scale Validation: NW vs SW across Similarity Levels '
                 '(50 pairs each, length 200)',
                 fontsize=12, fontweight='bold')

    # Panel 1 — Mean score
    ax = axes[0]
    ax.plot(sims, nw_scores, 'o-', color='#d62728', linewidth=2,
            markersize=6, label='NW (global)')
    ax.plot(sims, sw_scores, 's-', color='#1f77b4', linewidth=2,
            markersize=6, label='SW (local)')
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('True Similarity (%)', fontsize=10)
    ax.set_ylabel('Mean Alignment Score', fontsize=10)
    ax.set_title('Score', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2 — Mean identity %
    ax = axes[1]
    ax.plot(sims, nw_ids, 'o-', color='#d62728', linewidth=2,
            markersize=6, label='NW')
    ax.plot(sims, sw_ids, 's-', color='#1f77b4', linewidth=2,
            markersize=6, label='SW')
    ax.plot(sims, sims, ':', color='grey', linewidth=1.5,
            label='y = x (ideal)')
    ax.set_xlabel('True Similarity (%)', fontsize=10)
    ax.set_ylabel('Reported Identity (%)', fontsize=10)
    ax.set_title('Identity', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 3 — Mean gap rate %
    ax = axes[2]
    ax.plot(sims, nw_gaps, 'o-', color='#d62728', linewidth=2,
            markersize=6, label='NW')
    ax.plot(sims, sw_gaps, 's-', color='#1f77b4', linewidth=2,
            markersize=6, label='SW')
    ax.set_xlabel('True Similarity (%)', fontsize=10)
    ax.set_ylabel('Gap Rate (%)', fontsize=10)
    ax.set_title('Gap Rate', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved -> {save_path}')
