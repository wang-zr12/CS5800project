"""
main1.py — Extended experiments with real data and large-scale validation.

Section A:  SARS-CoV-2 spike RBD — NW vs SW core comparison on real sequences.
            4 methods (NW-linear, SW-linear, Gotoh-NW, Gotoh-SW) on 3 variant
            pairs (~585 nt).  No heatmaps (unreadable at this scale); output is
            alignment text + metrics + comparison table.

Section B:  Large-scale validation — controlled-similarity study.
            Generate 50 random pairs at each of 7 similarity levels
            (100%, 95%, 90%, 80%, 70%, 50%, 30%), length 200.
            Run NW and SW on every pair (700 total).  Test the key hypotheses:
              H1  NW == SW when similarity is 100%
              H2  SW score >= NW score always
              H3  SW advantage grows as similarity drops
              H4  NW identity degrades; SW identity stays high

Section C:  E-value / statistical significance.
            Karlin-Altschul parameters, E-values for all SARS-CoV-2 local
            alignments, plus a random-pair negative control.

Section D:  Multi-method timing benchmark.
            NW-linear, SW-linear, Gotoh-NW, Gotoh-SW, Biopython (optional).
            Log-scale timing chart.

All figures -> output/    All text results -> results/

Usage:
    python main1.py
"""

import math
import os
import random
import time
from collections import OrderedDict

from alignment import (
    needleman_wunsch, smith_waterman, ScoringParams,
    gotoh_global, gotoh_local, AffineScoringParams,
    needleman_wunsch_general, smith_waterman_general, GeneralGapParams,
)
from evalue import (
    compute_lambda, estimate_K,
    compute_evalue, compute_bit_score, interpret_evalue,
)
from sequences import get_sequences, PAIRWISE_COMPARISONS
from visualization import (
    format_alignment,
    compute_metrics,
    plot_timing_multi,
    plot_validation_study,
)

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), 'images')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'dataset_results')

_SEP  = '\u2500' * 76
_SEP2 = '\u2550' * 76


# ── helpers ──────────────────────────────────────────────────────────────────

def _write(path: str, text: str) -> None:
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(text)
    print(f'  Saved -> {path}')


def _metrics_header() -> str:
    return (f'  {"Method":<12s} {"Score":>6s} {"Len":>5s} '
            f'{"Ident":>7s} {"Gaps":>6s} {"Sc/Len":>6s} {"Cov":>6s}')


def _metrics_row(tag: str, m: dict) -> str:
    return (f'  {tag:<12s} {m["score"]:>6} {m["aligned_len"]:>5} '
            f'{m["identity_pct"]:>6.1f}% {m["gap_rate_pct"]:>5.1f}% '
            f'{m["score_density"]:>6.2f} {m["coverage_pct"]:>5.1f}%')


def _generate_pair(length: int, similarity: float, rng: random.Random):
    """
    Generate two sequences with a controlled nucleotide similarity level.

    `similarity` = 1.0 means identical; 0.25 means random (expected by chance
    on a 4-letter alphabet).
    """
    bases = 'ACGT'
    seq1 = [rng.choice(bases) for _ in range(length)]
    seq2 = list(seq1)
    n_mut = int(length * (1.0 - similarity))
    positions = rng.sample(range(length), min(n_mut, length))
    for pos in positions:
        alts = [b for b in bases if b != seq2[pos]]
        seq2[pos] = rng.choice(alts)
    return ''.join(seq1), ''.join(seq2)


# ═════════════════════════════════════════════════════════════════════════════
def run() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    linear  = ScoringParams(match=2, mismatch=-1, gap=-2)
    affine  = AffineScoringParams(match=2, mismatch=-1,
                                  gap_open=-10, gap_extend=-1)

    print(_SEP2)
    print('  Extended DNA Alignment — Real Data & Large-Scale Validation')
    print(_SEP2)
    print(f'  Linear : match={linear.match}  mismatch={linear.mismatch}'
          f'  gap={linear.gap}')
    print(f'  Affine : match={affine.match}  mismatch={affine.mismatch}'
          f'  gap_open={affine.gap_open}  gap_extend={affine.gap_extend}')
    print(_SEP2)
    print()

    # ─── Section A ───────────────────────────────────────────────────────
    # SARS-CoV-2 spike RBD: NW vs SW core comparison (no heatmaps)
    # ─────────────────────────────────────────────────────────────────────
    print(_SEP)
    print('  SECTION A — SARS-CoV-2 Spike RBD: NW vs SW (3 pairs × 4 methods)')
    print(_SEP)
    print()

    seqs = get_sequences(use_ncbi=False)
    all_results = []             # (pair_label, method_tag, AlignmentResult)
    txt_buf = [
        '# SARS-CoV-2 Spike RBD — NW vs SW Core Comparison\n'
        f'# 3 variant pairs, 4 methods, ~585 nt per sequence\n'
        f'# Linear: match={linear.match} mismatch={linear.mismatch} '
        f'gap={linear.gap}\n'
        f'# Affine: match={affine.match} mismatch={affine.mismatch} '
        f'gap_open={affine.gap_open} gap_extend={affine.gap_extend}\n\n'
    ]

    methods_to_run = [
        ('NW-linear', lambda s1, s2: needleman_wunsch(s1, s2, linear)),
        ('SW-linear', lambda s1, s2: smith_waterman(s1, s2, linear)),
        ('Gotoh-NW',  lambda s1, s2: gotoh_global(s1, s2, affine)),
        ('Gotoh-SW',  lambda s1, s2: gotoh_local(s1, s2, affine)),
    ]

    for key1, key2 in PAIRWISE_COMPARISONS:
        rec1, rec2 = seqs[key1], seqs[key2]
        pair = f'{rec1.id} vs {rec2.id}'

        print(f'  {pair}')
        print(f'    {rec1.accession} ({len(rec1.sequence)} nt) | '
              f'{rec2.accession} ({len(rec2.sequence)} nt)')
        print()

        txt_buf.append(f'{"=" * 70}\n{pair}\n{"=" * 70}\n')

        # Run all 4 methods
        pair_metrics = []
        for tag, func in methods_to_run:
            res = func(rec1.sequence, rec2.sequence)
            m   = compute_metrics(res)
            pair_metrics.append((tag, res, m))
            all_results.append((pair, tag, res))

            txt_buf.append(f'\n[{tag}]\n{format_alignment(res)}\n')

        # Compact comparison table
        print(_metrics_header())
        print('  ' + '-' * 70)
        for tag, res, m in pair_metrics:
            print(_metrics_row(tag, m))
        print()

        txt_buf.append('\n')

    _write(os.path.join(RESULTS_DIR, 'sars_alignments.txt'),
           ''.join(txt_buf))

    # Highlight the NW vs SW difference
    print('  Key observation:')
    print('    NW (global) aligns the full ~585 nt end-to-end, paying for')
    print('    every mismatch and gap across the entire length.')
    print('    SW (local) extracts only the best-matching contiguous region,')
    print('    ignoring low-quality flanking regions.')
    print('    On closely related variants (>97% identity), both give similar')
    print('    results.  The gap cost model (linear vs affine) matters more')
    print('    than the alignment scope when divergence is low.')
    print()

    # ─── Section B ───────────────────────────────────────────────────────
    # Large-scale validation: controlled-similarity study
    # ─────────────────────────────────────────────────────────────────────
    print(_SEP)
    print('  SECTION B — Large-Scale Validation (350 pairs × 2 algorithms)')
    print(_SEP)
    print()
    print('  Generating 50 random pairs at each of 7 similarity levels,')
    print('  length 200 nt.  Running NW and SW on every pair ...')
    print()

    sim_levels = [1.00, 0.95, 0.90, 0.80, 0.70, 0.50, 0.30]
    n_pairs    = 50
    pair_len   = 200
    rng        = random.Random(42)

    validation_data = []     # list of dicts for the plot
    val_txt = [
        '# Large-Scale Validation: NW vs SW\n'
        f'# {n_pairs} pairs per level, length {pair_len}, seed 42\n'
        f'# Scoring: match={linear.match} mismatch={linear.mismatch} '
        f'gap={linear.gap}\n\n'
        f'{"Sim%":>5s}  {"NW_score":>9s} {"SW_score":>9s} {"SW/NW":>7s}  '
        f'{"NW_ident":>9s} {"SW_ident":>9s}  '
        f'{"NW_gaps":>8s} {"SW_gaps":>8s}  '
        f'{"H2_pass":>7s}\n'
        + '-' * 88 + '\n'
    ]

    h2_violations = 0       # count times SW_score < NW_score

    for sim in sim_levels:
        nw_scores, sw_scores = [], []
        nw_ids, sw_ids       = [], []
        nw_gaps, sw_gaps     = [], []
        h2_ok = 0

        for _ in range(n_pairs):
            s1, s2 = _generate_pair(pair_len, sim, rng)
            r_nw = needleman_wunsch(s1, s2, linear)
            r_sw = smith_waterman(s1, s2, linear)
            m_nw = compute_metrics(r_nw)
            m_sw = compute_metrics(r_sw)

            nw_scores.append(r_nw.score)
            sw_scores.append(r_sw.score)
            nw_ids.append(m_nw['identity_pct'])
            sw_ids.append(m_sw['identity_pct'])
            nw_gaps.append(m_nw['gap_rate_pct'])
            sw_gaps.append(m_sw['gap_rate_pct'])

            if r_sw.score >= r_nw.score:
                h2_ok += 1
            else:
                h2_violations += 1

        nw_s = sum(nw_scores) / n_pairs
        sw_s = sum(sw_scores) / n_pairs
        ratio = sw_s / nw_s if nw_s != 0 else float('inf')

        row = {
            'similarity':       sim,
            'nw_score_mean':    nw_s,
            'sw_score_mean':    sw_s,
            'nw_identity_mean': sum(nw_ids) / n_pairs,
            'sw_identity_mean': sum(sw_ids) / n_pairs,
            'nw_gap_rate_mean': sum(nw_gaps) / n_pairs,
            'sw_gap_rate_mean': sum(sw_gaps) / n_pairs,
        }
        validation_data.append(row)

        pct = sim * 100
        print(f'    {pct:5.0f}%  NW={nw_s:>8.1f}  SW={sw_s:>8.1f}  '
              f'ratio={ratio:>5.2f}  '
              f'NW_ident={row["nw_identity_mean"]:>5.1f}%  '
              f'SW_ident={row["sw_identity_mean"]:>5.1f}%  '
              f'H2={h2_ok}/{n_pairs}')

        ratio_s = f'{ratio:.2f}' if nw_s != 0 else 'inf'
        val_txt.append(
            f'{pct:>5.0f}  {nw_s:>9.1f} {sw_s:>9.1f} {ratio_s:>7s}  '
            f'{row["nw_identity_mean"]:>8.1f}% {row["sw_identity_mean"]:>8.1f}%  '
            f'{row["nw_gap_rate_mean"]:>7.1f}% {row["sw_gap_rate_mean"]:>7.1f}%  '
            f'{h2_ok:>3d}/{n_pairs}\n'
        )

    print()
    print(f'  H2 violations (SW < NW): {h2_violations} / {n_pairs * len(sim_levels)}')
    print()

    val_txt.append(f'\nH2 violations (SW_score < NW_score): '
                   f'{h2_violations} / {n_pairs * len(sim_levels)}\n')
    _write(os.path.join(RESULTS_DIR, 'validation_study.txt'),
           ''.join(val_txt))

    plot_validation_study(
        validation_data,
        save_path=os.path.join(OUTPUT_DIR, 'validation_study.png'),
    )

    # ─── Section C ───────────────────────────────────────────────────────
    # E-value analysis (SARS-CoV-2 + random negative control)
    # ─────────────────────────────────────────────────────────────────────
    print(_SEP)
    print('  SECTION C — E-value / Statistical Significance')
    print(_SEP)
    print()

    print('  Computing lambda (bisection) ...')
    lam = compute_lambda(linear)
    print(f'    lambda = {lam:.8f}')

    print('  Estimating K (Monte Carlo, 2000 pairs, len 100) ...')
    K = estimate_K(linear, lam, seq_length=100, num_pairs=2000, seed=42)
    print(f'    K      = {K:.8f}')
    print()

    _write(os.path.join(RESULTS_DIR, 'karlin_altschul_params.txt'),
           f'# Karlin-Altschul Parameters\n'
           f'# Scoring: match={linear.match} mismatch={linear.mismatch} '
           f'gap={linear.gap}\n'
           f'# Base freqs: uniform 0.25\n'
           f'# K: Monte Carlo 2000 pairs len 100 seed 42\n\n'
           f'lambda = {lam:.10f}\nK      = {K:.10f}\n')

    # E-value table header
    hdr = (f'  {"Pair":<32s} {"Method":<12s} {"Score":>6s} '
           f'{"E-value":>12s} {"Bits":>8s}  Interpretation')
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))

    ev_lines = [
        f'# E-value Summary\n# lambda={lam:.8f}  K={K:.8f}\n\n'
        f'{"Pair":<32s} {"Method":<12s} {"Score":>6s} '
        f'{"E-value":>12s} {"Bits":>8s}  Interpretation\n'
        + '-' * 95 + '\n'
    ]

    # SW results from Section A
    for pair, tag, res in all_results:
        if 'SW' not in tag:
            continue                       # E-value applies to local only
        ev   = compute_evalue(res.score, len(res.seq1), len(res.seq2), lam, K)
        bits = compute_bit_score(res.score, lam, K)
        interp = interpret_evalue(ev)
        print(f'  {pair:<32s} {tag:<12s} {res.score:>6} '
              f'{ev:>12.2e} {bits:>8.1f}  {interp}')
        ev_lines.append(f'{pair:<32s} {tag:<12s} {res.score:>6} '
                        f'{ev:>12.2e} {bits:>8.1f}  {interp}\n')

    # Negative control: random pairs
    print()
    print('  --- Negative control: 5 random pairs (length 200) ---')
    ev_lines.append('\n--- Negative control: random pairs (len 200) ---\n')
    rng2 = random.Random(99)
    for i in range(5):
        s1 = ''.join(rng2.choices('ACGT', k=200))
        s2 = ''.join(rng2.choices('ACGT', k=200))
        r  = smith_waterman(s1, s2, linear)
        ev = compute_evalue(r.score, 200, 200, lam, K)
        bits = compute_bit_score(r.score, lam, K)
        interp = interpret_evalue(ev)
        label = f'random pair {i+1}'
        print(f'  {label:<32s} {"SW-linear":<12s} {r.score:>6} '
              f'{ev:>12.2e} {bits:>8.1f}  {interp}')
        ev_lines.append(f'{label:<32s} {"SW-linear":<12s} {r.score:>6} '
                        f'{ev:>12.2e} {bits:>8.1f}  {interp}\n')

    print()
    _write(os.path.join(RESULTS_DIR, 'evalue_table.txt'), ''.join(ev_lines))

    # ─── Section D ───────────────────────────────────────────────────────
    # Multi-method timing benchmark
    # ─────────────────────────────────────────────────────────────────────
    print(_SEP)
    print('  SECTION D — Multi-Method Timing Benchmark')
    print(_SEP)
    print()

    bench_methods: OrderedDict = OrderedDict()
    bench_methods['NW (linear)'] = lambda s1, s2: needleman_wunsch(s1, s2,
                                                                    linear)
    bench_methods['SW (linear)'] = lambda s1, s2: smith_waterman(s1, s2,
                                                                  linear)
    bench_methods['Gotoh-NW']    = lambda s1, s2: gotoh_global(s1, s2, affine)
    bench_methods['Gotoh-SW']    = lambda s1, s2: gotoh_local(s1, s2, affine)

    # General-gap O(n³): same scoring as linear gap for apples-to-apples
    _gen = GeneralGapParams.linear(gap=-2.0)
    bench_methods['General-NW'] = lambda s1, s2, p=_gen: needleman_wunsch_general(s1, s2, p)
    bench_methods['General-SW'] = lambda s1, s2, p=_gen: smith_waterman_general(s1, s2, p)

    try:
        from Bio.Align import PairwiseAligner
        _bp = PairwiseAligner()
        _bp.mode = 'local'
        _bp.match_score = 2; _bp.mismatch_score = -1
        _bp.open_gap_score = -10; _bp.extend_gap_score = -1
        bench_methods['Biopython'] = lambda s1, s2, a=_bp: a.score(s1, s2)
        print('  Biopython detected.')
    except ImportError:
        print('  (Biopython not installed — skipping)')
    print()

    # O(mn) methods: benchmark up to 1000 nt
    py_lengths  = [10, 25, 50, 100, 200, 300, 500, 750, 1000]
    # O(n³) general-gap: limited to avoid multi-minute wait
    gen_lengths = [10, 25, 50, 75, 100, 150, 200]
    # Biopython (C): extend to show magnitude of speedup
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
        for length in lengths:
            total = 0.0
            for _ in range(repeats):
                s1 = ''.join(random.choices('ACGT', k=length))
                s2 = ''.join(random.choices('ACGT', k=length))
                t0 = time.perf_counter()
                func(s1, s2)
                total += time.perf_counter() - t0
            timing_data[name].append((length, total / repeats))

    # Print table
    all_lens = sorted(set(l for pts in timing_data.values() for l, _ in pts))
    names    = list(timing_data.keys())
    cw = 12
    hdr = f'  {"Len":>6s}' + ''.join(f'  {n:>{cw}s}' for n in names)
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))

    t_lines = [f'# Multi-Method Timing (ms), {repeats} repeats, seed 42\n\n']
    t_lines.append(f'{"Len":>6s}' + ''.join(f'  {n:>{cw}s}' for n in names) + '\n')
    t_lines.append('-' * (6 + (cw + 2) * len(names)) + '\n')

    for ln in all_lens:
        row = f'  {ln:>6d}'
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
    _write(os.path.join(RESULTS_DIR, 'timing_methods.txt'), ''.join(t_lines))
    plot_timing_multi(timing_data,
                      save_path=os.path.join(OUTPUT_DIR, 'timing_methods.png'))

    # ─── Summary ─────────────────────────────────────────────────────────
    print()
    print(_SEP2)
    print('  Done.')
    print(f'  Sequences used: {len(seqs)} SARS-CoV-2 variants '
          f'+ {n_pairs * len(sim_levels)} synthetic pairs')
    print(f'  Figures  -> {OUTPUT_DIR}/')
    print(f'  Results  -> {RESULTS_DIR}/')
    print(_SEP2)
    print()


if __name__ == '__main__':
    run()
