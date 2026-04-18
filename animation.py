"""
animation.py — Animated GIF of the DP-fill and traceback process.

Generates one GIF per (experiment, algorithm) showing:
  Phase 1  cells filled one-by-one (or row-by-row for large matrices)
  Phase 2  traceback path drawn step-by-step
"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize

from alignment import AlignmentResult


_CMAPS = {
    'Needleman-Wunsch': 'YlOrRd',
    'Smith-Waterman':   'Blues',
    'Gotoh-NW':         'YlOrRd',
    'Gotoh-SW':         'Blues',
}


def create_fill_animation(
    result: AlignmentResult,
    save_path: str,
    interval_ms: int = 80,
    figsize: tuple = (9, 7.5),
) -> None:
    """
    Build an animated GIF that replays the DP fill then the traceback.
    """
    matrix = result.matrix.astype(float)
    seq1, seq2 = result.seq1, result.seq2
    m, n = len(seq1), len(seq2)

    # ── Masked display array (unfilled cells → grey) ─────────────────────
    display = ma.masked_all((m + 1, n + 1), dtype=float)

    # Border cells are visible from frame 0
    for i in range(m + 1):
        display[i, 0] = matrix[i, 0]
    for j in range(n + 1):
        display[0, j] = matrix[0, j]

    # ── Colour map ───────────────────────────────────────────────────────
    cmap_name = _CMAPS.get(result.algorithm, 'viridis')
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color='#e0e0e0')          # unfilled cells
    norm = Normalize(vmin=matrix.min(), vmax=matrix.max())

    # ── Figure ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(display, cmap=cmap, norm=norm, aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02)

    ax.set_xticks(range(n + 1))
    ax.set_xticklabels(['-'] + list(seq2), fontsize=9, fontfamily='monospace')
    ax.set_yticks(range(m + 1))
    ax.set_yticklabels(['-'] + list(seq1), fontsize=9, fontfamily='monospace')
    ax.set_title(result.algorithm, fontsize=12, fontweight='bold')

    # ── Pre-create text objects (hidden) ─────────────────────────────────
    show_text = (m + 1) * (n + 1) <= 300
    texts: dict = {}
    if show_text:
        vmin, vmax = float(matrix.min()), float(matrix.max())
        span = (vmax - vmin) if vmax != vmin else 1.0
        for i in range(m + 1):
            for j in range(n + 1):
                nv = (matrix[i, j] - vmin) / span
                c  = 'white' if nv > 0.55 else 'black'
                t  = ax.text(
                    j, i, str(int(matrix[i, j])),
                    ha='center', va='center',
                    fontsize=7, fontfamily='monospace',
                    color=c, fontweight='bold',
                )
                t.set_visible(i == 0 or j == 0)    # borders visible
                texts[(i, j)] = t

    # ── Artists ──────────────────────────────────────────────────────────
    (highlight,)  = ax.plot([], [], 's', color='yellow', markersize=14,
                            markeredgecolor='orange', markeredgewidth=2,
                            alpha=0.55, zorder=10)
    (trace_bg,)   = ax.plot([], [], '-', color='white', linewidth=4,
                            alpha=0.55, zorder=7)
    (trace_line,) = ax.plot([], [], '-', color='cyan', linewidth=2.2,
                            zorder=8)
    status = ax.text(0.5, -0.06, '', transform=ax.transAxes,
                     ha='center', fontsize=9, fontweight='bold')

    # ── Frame schedule ───────────────────────────────────────────────────
    fill_cells = [(i, j) for i in range(1, m + 1) for j in range(1, n + 1)]

    # Row-by-row for large matrices, cell-by-cell for small
    cell_by_cell = len(fill_cells) <= 200
    if cell_by_cell:
        fill_frames = fill_cells                           # list[(i,j)]
    else:
        fill_frames = []
        for row in range(1, m + 1):
            fill_frames.append([(row, j) for j in range(1, n + 1)])

    path       = result.traceback_path
    n_fill     = len(fill_frames)
    n_pause    = 8
    n_trace    = len(path)
    n_hold     = 15
    total      = n_fill + n_pause + n_trace + n_hold

    # ── Update function ──────────────────────────────────────────────────
    def update(frame: int):
        if frame < n_fill:
            # --- Fill phase ---
            if cell_by_cell:
                i, j = fill_frames[frame]
                display[i, j] = matrix[i, j]
                highlight.set_data([j], [i])
                if (i, j) in texts:
                    texts[(i, j)].set_visible(True)
                status.set_text(f'Fill ({i},{j}) = {int(matrix[i, j])}')
            else:
                cells = fill_frames[frame]
                for (i, j) in cells:
                    display[i, j] = matrix[i, j]
                    if (i, j) in texts:
                        texts[(i, j)].set_visible(True)
                ri, rj = cells[-1]
                highlight.set_data([rj], [ri])
                status.set_text(f'Fill row {cells[0][0]}')
            im.set_data(display)

        elif frame < n_fill + n_pause:
            # --- Pause ---
            highlight.set_data([], [])
            status.set_text('Fill complete \u2014 starting traceback\u2026')

        elif frame < n_fill + n_pause + n_trace:
            # --- Traceback phase ---
            step = frame - n_fill - n_pause
            pts  = path[:step + 1]
            rows = [p[0] for p in pts]
            cols = [p[1] for p in pts]
            trace_bg.set_data(cols, rows)
            trace_line.set_data(cols, rows)
            highlight.set_data([cols[-1]], [rows[-1]])
            status.set_text(f'Traceback step {step + 1}/{n_trace}')

        else:
            # --- Hold ---
            highlight.set_data([], [])
            # Add start / end markers on final hold
            if path:
                ax.plot(path[0][1], path[0][0], 'o', color='#2ca02c',
                        markersize=8, zorder=11,
                        path_effects=[pe.withStroke(linewidth=1.8,
                                                    foreground='white')])
                ax.plot(path[-1][1], path[-1][0], 's', color='#d62728',
                        markersize=8, zorder=11,
                        path_effects=[pe.withStroke(linewidth=1.8,
                                                    foreground='white')])
            status.set_text('Complete')

        return [im, highlight, trace_bg, trace_line, status]

    # ── Build and save ───────────────────────────────────────────────────
    anim = FuncAnimation(fig, update, frames=total,
                         interval=interval_ms, blit=False)
    fps  = max(1, 1000 // interval_ms)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f'  Saved animation -> {save_path}')
