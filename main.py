"""
main.py — Entry point.

Runs five experiments, outputs one PNG per (experiment, algorithm),
a score-comparison chart, a timing benchmark, and DP-fill animations.

Usage:
    python main.py
"""

import os
import time
import random

from alignment import needleman_wunsch, smith_waterman, ScoringParams
from experiments import EXPERIMENTS
from visualization import (
    format_alignment,
    compute_metrics,
    plot_single_matrix,
    plot_score_comparison,
    plot_timing,
)
from animation import create_fill_animation

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'images')

_SEP  = '\u2500' * 64
_SEP2 = '\u2550' * 64


# ── helpers ──────────────────────────────────────────────────────────────────

def _metrics_row(tag: str, m: dict) -> str:
    return (
        f'  {tag:<6s}  {m["score"]:>5}  {m["aligned_len"]:>5}  '
        f'{m["identity_pct"]:>6.1f}%  {m["gap_rate_pct"]:>5.1f}%  '
        f'{m["score_density"]:>6.2f}  {m["coverage_pct"]:>6.1f}%'
    )


def _metrics_header() -> str:
    return (
        f'  {"":6s}  {"Score":>5s}  {"Len":>5s}  '
        f'{"Ident":>7s}  {"Gaps":>6s}  '
        f'{"Sc/Len":>6s}  {"Cov":>7s}'
    )


# ── main ─────────────────────────────────────────────────────────────────────

def run() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    params = ScoringParams(match=2, mismatch=-1, gap=-2)

    print(_SEP2)
    print('  DNA Alignment: Smith-Waterman  vs  Needleman-Wunsch')
    print(_SEP2)
    print(f'  Scoring:  match={params.match}  mismatch={params.mismatch}'
          f'  gap={params.gap}')
    print(_SEP2)
    print()

    scores_data = []

    # ── Run each experiment ──────────────────────────────────────────────
    for idx, exp in enumerate(EXPERIMENTS, start=1):
        seq1 = exp['seq1']
        seq2 = exp['seq2']
        eid  = exp['id']
        name = exp['name']

        print(_SEP)
        print(f'  Experiment {idx}: {name}')
        print(f'  Seq1 ({len(seq1):>2d}): {seq1}')
        print(f'  Seq2 ({len(seq2):>2d}): {seq2}')
        print()

        nw = needleman_wunsch(seq1, seq2, params)
        sw = smith_waterman(seq1, seq2, params)

        # Formatted alignment text
        print('  -- Needleman-Wunsch (Global) --')
        for line in format_alignment(nw).split('\n'):
            print(f'     {line}')
        print()
        print('  -- Smith-Waterman (Local) --')
        for line in format_alignment(sw).split('\n'):
            print(f'     {line}')
        print()

        # Extended metrics
        nw_m = compute_metrics(nw)
        sw_m = compute_metrics(sw)
        print(_metrics_header())
        print(f'  {"":-<6s}  {"-----":>5s}  {"-----":>5s}  '
              f'{"-------":>7s}  {"------":>6s}  '
              f'{"------":>6s}  {"-------":>7s}')
        print(_metrics_row('NW', nw_m))
        print(_metrics_row('SW', sw_m))
        print()

        scores_data.append({
            'name': name,
            'nw_score': nw.score,
            'sw_score': sw.score,
        })

        # Individual matrix PNGs
        plot_single_matrix(nw, os.path.join(OUTPUT_DIR, f'{eid}_NW.png'))
        plot_single_matrix(sw, os.path.join(OUTPUT_DIR, f'{eid}_SW.png'))

    # ── Score comparison chart ───────────────────────────────────────────
    print(_SEP)
    print()
    plot_score_comparison(
        scores_data,
        save_path=os.path.join(OUTPUT_DIR, 'score_comparison.png'),
    )

    # ── Timing benchmark ─────────────────────────────────────────────────
    print()
    print('  Running timing benchmark ...')
    lengths = [10, 25, 50, 100, 150, 200, 300, 400, 500]
    repeats = 3
    timing_data = []

    random.seed(42)
    for length in lengths:
        nw_total = 0.0
        sw_total = 0.0
        for _ in range(repeats):
            s1 = ''.join(random.choices('ACGT', k=length))
            s2 = ''.join(random.choices('ACGT', k=length))

            t0 = time.perf_counter()
            needleman_wunsch(s1, s2, params)
            nw_total += time.perf_counter() - t0

            t0 = time.perf_counter()
            smith_waterman(s1, s2, params)
            sw_total += time.perf_counter() - t0

        timing_data.append((length, nw_total / repeats, sw_total / repeats))

    print(f'\n  {"Len":>5s}   {"NW (ms)":>8s}   {"SW (ms)":>8s}')
    print(f'  {"-----":>5s}   {"--------":>8s}   {"--------":>8s}')
    for ln, nw_t, sw_t in timing_data:
        print(f'  {ln:5d}   {nw_t*1000:8.2f}   {sw_t*1000:8.2f}')

    plot_timing(timing_data, os.path.join(OUTPUT_DIR, 'timing_benchmark.png'))

    # ── Animations (Experiment 2: Partial Similarity) ────────────────────
    print()
    print('  Generating DP-fill animations for Experiment 2 ...')
    exp2 = EXPERIMENTS[1]
    nw2  = needleman_wunsch(exp2['seq1'], exp2['seq2'], params)
    sw2  = smith_waterman(exp2['seq1'], exp2['seq2'], params)

    create_fill_animation(nw2,
                          os.path.join(OUTPUT_DIR, 'partial_sim_NW_anim.gif'),
                          interval_ms=80)
    create_fill_animation(sw2,
                          os.path.join(OUTPUT_DIR, 'partial_sim_SW_anim.gif'),
                          interval_ms=80)

    # ── Summary table ────────────────────────────────────────────────────
    print()
    print(_SEP2)
    print('  Algorithmic Summary')
    print(_SEP2)
    rows = [
        ('Initialization',   'border = i*gap / j*gap',   'all borders = 0'),
        ('Recurrence floor', 'none (can go negative)',    'max(0, ...)'),
        ('Traceback start',  '(m, n) bottom-right',       'argmax(H)'),
        ('Traceback stop',   '(0, 0) top-left',           'H[i,j] = 0'),
        ('Negative score?',  'yes',                        'no'),
        ('Complexity',       'O(mn) time / O(mn) space',  'O(mn) time / O(mn) space'),
        ('Best use case',    'full-length similar seqs',   'conserved subregion search'),
    ]
    col = (22, 32, 32)
    print(f"  {'Property':<{col[0]}} {'Needleman-Wunsch':<{col[1]}} {'Smith-Waterman':<{col[2]}}")
    print('  ' + '-' * (sum(col) + 2))
    for prop, nw_val, sw_val in rows:
        print(f"  {prop:<{col[0]}} {nw_val:<{col[1]}} {sw_val:<{col[2]}}")
    print()
    print(f'  All outputs -> {OUTPUT_DIR}/')
    print()


if __name__ == '__main__':
    run()
