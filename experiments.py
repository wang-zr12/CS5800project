"""
experiments.py — Five test cases designed to expose the behavioural difference
between global (NW) and local (SW) alignment across increasingly challenging
scenarios.
"""

EXPERIMENTS = [
    # ── Experiment 1 ──────────────────────────────────────────────────────
    {
        'id': 'full_sim',
        'name': 'Full Similarity',
        'seq1': 'ACGTACGT',
        'seq2': 'ACGTACGT',
        'description': (
            'Identical sequences — baseline. Both algorithms should produce '
            'the same perfect alignment with score = 8 * match(2) = 16.'
        ),
    },
    # ── Experiment 2 ──────────────────────────────────────────────────────
    {
        'id': 'partial_sim',
        'name': 'Partial Similarity',
        'seq1': 'TTTACGTTTT',
        'seq2': 'GGGACGTGGG',
        'description': (
            'Shared core ACGT flanked by completely mismatching regions. '
            'SW isolates the core; NW absorbs flanking mismatch penalties.'
        ),
    },
    # ── Experiment 3 ──────────────────────────────────────────────────────
    {
        'id': 'shifted',
        'name': 'Shifted / Overlapping',
        'seq1': 'AATCGTAGCA',
        'seq2': 'CGTAGCATTG',
        'description': (
            'Shared substring CGTAGCA at opposing ends. '
            'SW finds it cleanly; NW pays heavy gap penalties.'
        ),
    },
    # ── Experiment 4 ──────────────────────────────────────────────────────
    {
        'id': 'embed_motif',
        'name': 'Embedded Motif (EcoRI)',
        'seq1': 'AACCTTGAATTCCGGAA',
        'seq2': 'TTTTGAATTCTTTT',
        'description': (
            'Both sequences carry the EcoRI restriction site GAATTC buried '
            'inside unrelated flanking nucleotides. Tests motif-level detection.'
        ),
    },
    # ── Experiment 5 ──────────────────────────────────────────────────────
    {
        'id': 'diff_length',
        'name': 'Different Lengths',
        'seq1': 'ATCGATCG',
        'seq2': 'TTTTATCGATCGTTTTT',
        'description': (
            'A short query embedded in a much longer target. '
            'SW finds the exact match; NW must pad with many gaps.'
        ),
    },
]
