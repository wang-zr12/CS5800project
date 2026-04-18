"""
run_timing.py — Standalone timing benchmark for ALL alignment methods.

Benchmarks 7 methods on synthetic random DNA pairs of increasing length,
produces a log-scale timing chart, and saves results.

Methods (3 complexity classes):
  O(mn)          NW-linear, SW-linear, Gotoh-NW, Gotoh-SW
  O(m²n+mn²)    General-NW, General-SW  (arbitrary gap function)
  O(mn) C-level  Biopython PairwiseAligner (optional)

Usage:
    python run_timing.py
"""

import os
import random
import time
from collections import OrderedDict

from alignment import (
    needleman_wunsch, smith_waterman, ScoringParams,
    gotoh_global, gotoh_local, AffineScoringParams,
    needleman_wunsch_general, smith_waterman_general, GeneralGapParams,
)
from visualization import plot_timing_multi

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), 'images')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'dataset_results')

_SEP  = '\u2500' * 76
_SEP2 = '\u2550' * 76


def run() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    linear = ScoringParams(match=2, mismatch=-1, gap=-2)
    affine = AffineScoringParams(match=2, mismatch=-1,
                                 gap_open=-10, gap_extend=-1)

    print(_SEP2)
    print('  Multi-Method Timing Benchmark (synthetic data)')
    print(_SEP2)
    print(f'  Linear : match={linear.match}  mismatch={linear.mismatch}'
          f'  gap={linear.gap}')
    print(f'  Affine : match={affine.match}  mismatch={affine.mismatch}'
          f'  gap_open={affine.gap_open}  gap_extend={affine.gap_extend}')
    print(_SEP2)
    print()

    # ── Register methods ─────────────────────────────────────────────────
    bench_methods: OrderedDict = OrderedDict()
    bench_methods['NW (linear)'] = lambda s1, s2: needleman_wunsch(s1, s2,
                                                                    linear)
    bench_methods['SW (linear)'] = lambda s1, s2: smith_waterman(s1, s2,
                                                                  linear)
    bench_methods['Gotoh-NW']    = lambda s1, s2: gotoh_global(s1, s2, affine)
    bench_methods['Gotoh-SW']    = lambda s1, s2: gotoh_local(s1, s2, affine)

    # General-gap O(n^3): use linear gap for apples-to-apples timing
    _gen = GeneralGapParams.linear(gap=-2.0)
    bench_methods['General-NW'] = \
        lambda s1, s2, p=_gen: needleman_wunsch_general(s1, s2, p)
    bench_methods['General-SW'] = \
        lambda s1, s2, p=_gen: smith_waterman_general(s1, s2, p)

    try:
        from Bio.Align import PairwiseAligner
        _bp = PairwiseAligner()
        _bp.mode = 'local'
        _bp.match_score = 2;  _bp.mismatch_score = -1
        _bp.open_gap_score = -10; _bp.extend_gap_score = -1
        bench_methods['Biopython'] = lambda s1, s2, a=_bp: a.score(s1, s2)
        print('  Biopython detected.')
    except ImportError:
        print('  (Biopython not installed - skipping)')
    print()

    # ── Length schedules per complexity class ─────────────────────────────
    py_lengths  = [10, 25, 50, 100, 200, 300, 500, 750, 1000]
    gen_lengths = [10, 25, 50, 75, 100, 150, 200]
    bp_extras   = [1500, 2000, 3000, 5000]
    repeats     = 3

    timing_data: OrderedDict = OrderedDict()
    random.seed(42)

    for name, func in bench_methods.items():
        timing_data[name] = []
        if name.startswith('General'):
            lengths = gen_lengths
        elif name == 'Biopython':
            lengths = py_lengths + bp_extras
        else:
            lengths = list(py_lengths)

        print(f'  Benchmarking {name} ...', end='', flush=True)
        for length in lengths:
            total = 0.0
            for _ in range(repeats):
                s1 = ''.join(random.choices('ACGT', k=length))
                s2 = ''.join(random.choices('ACGT', k=length))
                t0 = time.perf_counter()
                func(s1, s2)
                total += time.perf_counter() - t0
            timing_data[name].append((length, total / repeats))
        print(' done')

    print()

    # ── Print table ──────────────────────────────────────────────────────
    all_lens = sorted(set(l for pts in timing_data.values() for l, _ in pts))
    names    = list(timing_data.keys())
    cw = 12
    hdr = f'  {"Len":>6s}' + ''.join(f'  {n:>{cw}s}' for n in names)
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))

    t_lines = [f'# Multi-Method Timing (ms), {repeats} repeats, seed 42\n\n']
    t_lines.append(
        f'{"Len":>6s}' + ''.join(f'  {n:>{cw}s}' for n in names) + '\n')
    t_lines.append('-' * (6 + (cw + 2) * len(names)) + '\n')

    for ln in all_lens:
        row  = f'  {ln:>6d}'
        trow = f'{ln:>6d}'
        for n in names:
            pts = dict(timing_data[n])
            if ln in pts:
                ms = pts[ln] * 1000
                row  += f'  {ms:>{cw}.2f}'
                trow += f'  {ms:>{cw}.2f}'
            else:
                row  += f'  {"--":>{cw}s}'
                trow += f'  {"--":>{cw}s}'
        print(row)
        t_lines.append(trow + '\n')

    print()

    result_path = os.path.join(RESULTS_DIR, 'timing_methods.txt')
    with open(result_path, 'w', encoding='utf-8') as fh:
        fh.write(''.join(t_lines))
    print(f'  Saved -> {result_path}')

    chart_path = os.path.join(OUTPUT_DIR, 'timing_methods.png')
    plot_timing_multi(timing_data, save_path=chart_path)

    print()
    print(_SEP2)
    print('  Done.')
    print(f'  Chart  -> {chart_path}')
    print(f'  Table  -> {result_path}')
    print(_SEP2)


if __name__ == '__main__':
    run()
