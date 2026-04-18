#!/usr/bin/env python3
"""
comparison.py — Comprehensive comparison of sequence alignment algorithms.

Compares four alignment algorithms:
1. Needleman-Wunsch (linear gap penalty, global)
2. Smith-Waterman (linear gap penalty, local)
3. Gotoh-NW (affine gap penalty, global)
4. Gotoh-SW (affine gap penalty, local)

Tests on synthetic data and real SARS-CoV-2 sequences.
"""

import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib as mpl
import json
from typing import List, Dict, Tuple
from Bio import SeqIO
import warnings

warnings.filterwarnings('ignore')

# Import our professional alignment implementations
from alignment import (
    needleman_wunsch, smith_waterman, gotoh_global, gotoh_local,
    ScoringParams, AffineScoringParams, AlignmentResult,
    calculate_alignment_stats, format_alignment_display, list_algorithms
)


class AlignmentComparator:
    """Comprehensive alignment algorithm comparator."""

    def __init__(self, output_dir: str = "alignment_comparison"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Standard scoring parameters
        self.linear_params = ScoringParams(match=2, mismatch=-1, gap=-2)
        self.affine_params = AffineScoringParams(match=2, mismatch=-1, gap_open=-10, gap_extend=-1)

        # Algorithm registry
        self.algorithms = {
            'Needleman-Wunsch': lambda s1, s2: needleman_wunsch(s1, s2, self.linear_params),
            'Smith-Waterman': lambda s1, s2: smith_waterman(s1, s2, self.linear_params),
            'Gotoh-NW': lambda s1, s2: gotoh_global(s1, s2, self.affine_params),
            'Gotoh-SW': lambda s1, s2: gotoh_local(s1, s2, self.affine_params)
        }

    def run_test_cases(self) -> List[Dict]:
        """Run comprehensive test cases comparing all algorithms."""
        print("Running algorithm test cases...")

        # Define comprehensive test cases
        test_cases = [
            {
                "name": "Perfect match",
                "seq1": "ATCGATCG",
                "seq2": "ATCGATCG",
                "description": "Two identical sequences",
                "expected": "All algorithms should produce the same result"
            },
            {
                "name": "Single substitution",
                "seq1": "ATCGATCG",
                "seq2": "ATCGCTCG",
                "description": "Single point substitution (A->C)",
                "expected": "Global algorithms may score higher"
            },
            {
                "name": "Insertion mutation",
                "seq1": "ATCGATCG",
                "seq2": "ATCGAATCG",
                "description": "Insertion of a single 'A'",
                "expected": "Gotoh algorithms handle gaps better"
            },
            {
                "name": "Deletion mutation",
                "seq1": "ATCGATCG",
                "seq2": "ATCGTCG",
                "description": "Deletion of a single 'A'",
                "expected": "Gotoh algorithms handle gaps better"
            },
            {
                "name": "Multiple gaps",
                "seq1": "ATCGATCGAA",
                "seq2": "AT--ATC--A",
                "description": "Multiple gap regions",
                "expected": "Linear gap penalties may over-penalize"
            },
            {
                "name": "Long insertion",
                "seq1": "ATCG",
                "seq2": "ATCGATCGATCG",
                "description": "Long inserted sequence",
                "expected": "Local algorithms may perform better"
            },
            {
                "name": "Local similarity",
                "seq1": "AAATCGATCGAAA",
                "seq2": "TTTATCGATCGTTT",
                "description": "Middle region is similar",
                "expected": "Smith-Waterman and Gotoh-SW show advantages"
            },
            {
                "name": "Repeated sequence",
                "seq1": "ATCGATCGATCG",
                "seq2": "ATCGATCG",
                "description": "One sequence is a repeat of the other",
                "expected": "Local algorithms find repeating units"
            },
            {
                "name": "No similarity",
                "seq1": "AAAAAAA",
                "seq2": "TTTTTTT",
                "description": "Completely different sequences",
                "expected": "All algorithms should score very low"
            },
            {
                "name": "Large length difference",
                "seq1": "AT",
                "seq2": "ATCGATCGATCGATCGATCG",
                "description": "Sequence length differs by an order of magnitude",
                "expected": "Local algorithms are more stable"
            }
        ]

        results = []

        for test_case in test_cases:
            print(f"\nTest case: {test_case['name']}")
            print(f"Sequence 1: {test_case['seq1']}")
            print(f"Sequence 2: {test_case['seq2']}")
            print(f"Expected: {test_case['expected']}")

            case_results = {
                'test_case': test_case['name'],
                'description': test_case['description'],
                'expected': test_case['expected'],
                'seq1': test_case['seq1'],
                'seq2': test_case['seq2'],
                'algorithm_results': {}
            }

            # Run all algorithms
            for alg_name, alg_func in self.algorithms.items():
                start_time = time.time()
                try:
                    result = alg_func(test_case['seq1'], test_case['seq2'])
                    execution_time = time.time() - start_time

                    # Calculate detailed statistics
                    stats = calculate_alignment_stats(result.seq1_aligned, result.seq2_aligned)

                    case_results['algorithm_results'][alg_name] = {
                        'result': result,
                        'stats': stats,
                        'execution_time': execution_time
                    }

                    print(f"{alg_name:15} | Score: {result.score:6.1f} | "
                          f"Identity: {stats['identity_percent']:5.1f}% | "
                          f"Time: {execution_time * 1000:6.2f}ms")

                except Exception as e:
                    print(f"{alg_name:15} | Error: {str(e)}")
                    case_results['algorithm_results'][alg_name] = None

            results.append(case_results)
            print("-" * 80)

        return results

    def run_real_sequence_comparison(self, sequence_files: List[str]) -> List[Dict]:
        """Compare algorithm performance on real SARS-CoV-2 sequences; prefer Gotoh for analysis."""
        print("\nComparing algorithms on real SARS-CoV-2 sequences...")

        # For real data, prioritize Gotoh algorithms as recommended
        real_data_algorithms = {
            'Gotoh-NW': lambda s1, s2: gotoh_global(s1, s2, self.affine_params),
            'Gotoh-SW': lambda s1, s2: gotoh_local(s1, s2, self.affine_params),
            'Needleman-Wunsch': lambda s1, s2: needleman_wunsch(s1, s2, self.linear_params),
            'Smith-Waterman': lambda s1, s2: smith_waterman(s1, s2, self.linear_params)
        }

        # Collect sequences
        sequences = []
        for pattern in sequence_files:
            for file_path in glob.glob(pattern):
                try:
                    for record in SeqIO.parse(file_path, "fasta"):
                        seq_str = str(record.seq)
                        if len(seq_str) > 100:  # Only process sufficiently long sequences
                            sequences.append({
                                'id': record.id,
                                'seq': seq_str[:800],  # Limit length to avoid excessive computation
                                'file': os.path.basename(file_path),
                                'variant': self._identify_variant(record.id, record.description)
                            })
                        if len(sequences) >= 15:  # Limit total sequences
                            break
                except Exception as e:
                    print(f"Warning: Unable to read file {file_path}: {e}")
                    continue
                if len(sequences) >= 15:
                    break
            if len(sequences) >= 15:
                break

        if len(sequences) < 2:
            print("Warning: At least 2 sequences are required for comparison")
            return []

        print(f"Collected {len(sequences)} sequences, starting pairwise comparisons...")
        print("🧬 For real data, prefer Gotoh algorithms for higher-quality alignments")

        comparisons = []
        max_comparisons = 12  # Limit comparisons to avoid excessive computation
        count = 0

        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                if count >= max_comparisons:
                    break

                seq1 = sequences[i]
                seq2 = sequences[j]

                print(f"Comparison #{count + 1}: {seq1['variant']} vs {seq2['variant']}")

                comparison_result = {
                    'comparison_id': count + 1,
                    'seq1': seq1,
                    'seq2': seq2,
                    'same_variant': seq1['variant'] == seq2['variant'],
                    'algorithm_results': {}
                }

                # Run prioritized algorithms for real data
                for alg_name, alg_func in real_data_algorithms.items():
                    start_time = time.time()
                    try:
                        result = alg_func(seq1['seq'], seq2['seq'])
                        execution_time = time.time() - start_time

                        stats = calculate_alignment_stats(result.seq1_aligned, result.seq2_aligned)

                        comparison_result['algorithm_results'][alg_name] = {
                            'result': result,
                            'stats': stats,
                            'execution_time': execution_time
                        }

                        # Highlight Gotoh results
                        if alg_name.startswith('Gotoh'):
                            print(
                                f"  🎯 {alg_name:12}: {stats['identity_percent']:5.1f}% identity, {execution_time * 1000:6.1f}ms ⭐")
                        else:
                            print(
                                f"    {alg_name:12}: {stats['identity_percent']:5.1f}% identity, {execution_time * 1000:6.1f}ms")

                    except Exception as e:
                        print(f"  {alg_name} failed: {str(e)}")
                        comparison_result['algorithm_results'][alg_name] = None


                comparisons.append(comparison_result)
                count += 1

                # Show brief Gotoh-focused summary
                gotoh_nw = comparison_result['algorithm_results'].get('Gotoh-NW')
                gotoh_sw = comparison_result['algorithm_results'].get('Gotoh-SW')
                if gotoh_nw and gotoh_sw:
                    print(f"  📊 Gotoh comparison: global={gotoh_nw['stats']['identity_percent']:.1f}%, "
                          f"local={gotoh_sw['stats']['identity_percent']:.1f}%")

            if count >= max_comparisons:
                break

        print(f"Completed {len(comparisons)} pairwise comparisons")
        print("Tip: Gotoh algorithms provide more accurate gap modeling on real data")
        return comparisons

    def _identify_variant(self, seq_id: str, description: str) -> str:
        """Identify variant from sequence ID or description."""
        text = (seq_id + " " + description).lower()

        variant_keywords = {
            "Alpha": ["alpha", "b.1.1.7", "b117"],
            "Beta": ["beta", "b.1.351", "b1351"],
            "Gamma": ["gamma", "p.1", "p1"],
            "Delta": ["delta", "b.1.617.2", "b16172"],
            "Omicron": ["omicron", "b.1.1.529", "ba.1", "ba.2", "ba.4", "ba.5"]
        }

        for variant, keywords in variant_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return variant

        if any(term in text for term in ["wuhan", "original", "nc_045512", "reference"]):
            return "Original"

        return "Unknown"

    def analyze_algorithm_performance(self, test_results: List[Dict],
                                      real_results: List[Dict]) -> Dict:
        """Analyze algorithm performance."""
        print("Analyzing algorithm performance...")

        analysis = {
            'test_case_analysis': {},
            'real_sequence_analysis': {},
            'overall_comparison': {}
        }

        # Analyze test case results
        if test_results:
            alg_scores = {alg: [] for alg in self.algorithms.keys()}
            alg_identities = {alg: [] for alg in self.algorithms.keys()}
            alg_times = {alg: [] for alg in self.algorithms.keys()}

            for test_result in test_results:
                for alg_name, alg_result in test_result['algorithm_results'].items():
                    if alg_result:
                        alg_scores[alg_name].append(alg_result['result'].score)
                        alg_identities[alg_name].append(alg_result['stats']['identity_percent'])
                        alg_times[alg_name].append(alg_result['execution_time'])

            for alg_name in self.algorithms.keys():
                if alg_scores[alg_name]:
                    analysis['test_case_analysis'][alg_name] = {
                        'avg_score': np.mean(alg_scores[alg_name]),
                        'std_score': np.std(alg_scores[alg_name]),
                        'avg_identity': np.mean(alg_identities[alg_name]),
                        'std_identity': np.std(alg_identities[alg_name]),
                        'avg_time_ms': np.mean(alg_times[alg_name]) * 1000,
                        'std_time_ms': np.std(alg_times[alg_name]) * 1000
                    }

        # Analyze real sequence results
        if real_results:
            real_alg_scores = {alg: [] for alg in self.algorithms.keys()}
            real_alg_identities = {alg: [] for alg in self.algorithms.keys()}
            real_alg_times = {alg: [] for alg in self.algorithms.keys()}

            for comparison in real_results:
                for alg_name, alg_result in comparison['algorithm_results'].items():
                    if alg_result:
                        real_alg_scores[alg_name].append(alg_result['result'].score)
                        real_alg_identities[alg_name].append(alg_result['stats']['identity_percent'])
                        real_alg_times[alg_name].append(alg_result['execution_time'])

            for alg_name in self.algorithms.keys():
                if real_alg_scores[alg_name]:
                    analysis['real_sequence_analysis'][alg_name] = {
                        'avg_score': np.mean(real_alg_scores[alg_name]),
                        'std_score': np.std(real_alg_scores[alg_name]),
                        'avg_identity': np.mean(real_alg_identities[alg_name]),
                        'std_identity': np.std(real_alg_identities[alg_name]),
                        'avg_time_ms': np.mean(real_alg_times[alg_name]) * 1000,
                        'std_time_ms': np.std(real_alg_times[alg_name]) * 1000
                    }

        return analysis

    def visualize_comparison_results(self, test_results: List[Dict],
                                     real_results: List[Dict],
                                     analysis: Dict):
        """Visualize comparison results."""
        print("Generating comparison visualizations...")

        # ---- Ensure matplotlib uses a font that supports CJK characters if available ----
        preferred_fonts = [
            'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'NotoSansCJKsc-Regular',
            'WenQuanYi Zen Hei', 'Arial Unicode MS'
        ]
        chosen_font = None
        for fname in preferred_fonts:
            if any(fname == f.name or fname in f.name for f in fm.fontManager.ttflist):
                chosen_font = fname
                break

        if chosen_font:
            mpl.rcParams['font.sans-serif'] = [chosen_font]
            mpl.rcParams['font.family'] = 'sans-serif'
        else:
            # If no suitable font is found, warn the user. On Windows install 'Microsoft YaHei' or 'SimHei',
            # or install an appropriate Noto Sans CJK font on other OSes.
            print("Warning: No CJK-capable font found. Chinese text may render as boxes. Install 'Microsoft YaHei', 'SimHei', or 'Noto Sans CJK'.")

        # Ensure minus sign and other symbols render correctly
        mpl.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Comprehensive comparison of sequence alignment algorithms: NW vs SW vs Gotoh-NW vs Gotoh-SW',
                     fontsize=16, fontweight='bold')

        algorithms = list(self.algorithms.keys())
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

        # 1. Test case scores comparison
        if test_results and analysis.get('test_case_analysis'):
            test_names = [t['test_case'] for t in test_results[:8]]  # Limit to 8 for readability

            x = np.arange(len(test_names))
            width = 0.2

            for i, alg_name in enumerate(algorithms):
                if alg_name in analysis['test_case_analysis']:
                    scores = []
                    for test_result in test_results[:8]:
                        alg_result = test_result['algorithm_results'].get(alg_name)
                        if alg_result:
                            scores.append(alg_result['result'].score)
                        else:
                            scores.append(0)

                    axes[0, 0].bar(x + i * width, scores, width, label=alg_name,
                                   alpha=0.8, color=colors[i])

            axes[0, 0].set_xlabel('Test case')
            axes[0, 0].set_ylabel('Alignment score')
            axes[0, 0].set_title('Test case score comparison')
            axes[0, 0].set_xticks(x + width * 1.5)
            axes[0, 0].set_xticklabels(test_names, rotation=45, ha='right')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Identity percentage comparison
        if test_results:
            identity_data = []
            for alg_name in algorithms:
                identities = []
                for test_result in test_results:
                    alg_result = test_result['algorithm_results'].get(alg_name)
                    if alg_result:
                        identities.append(alg_result['stats']['identity_percent'])
                if identities:
                    identity_data.append(identities)
                else:
                    identity_data.append([0])

            axes[0, 1].boxplot(identity_data, tick_labels=algorithms)
            axes[0, 1].set_ylabel('Identity (%)')
            axes[0, 1].set_title('Identity distribution comparison')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Execution time comparison
        if analysis.get('test_case_analysis'):
            alg_names = []
            avg_times = []
            std_times = []

            for alg_name in algorithms:
                if alg_name in analysis['test_case_analysis']:
                    alg_names.append(alg_name)
                    avg_times.append(analysis['test_case_analysis'][alg_name]['avg_time_ms'])
                    std_times.append(analysis['test_case_analysis'][alg_name]['std_time_ms'])

            bars = axes[0, 2].bar(alg_names, avg_times, yerr=std_times,
                                  capsize=5, alpha=0.8, color=colors[:len(alg_names)])
            axes[0, 2].set_ylabel('Execution time (ms)')
            axes[0, 2].set_title('Average execution time comparison')
            axes[0, 2].tick_params(axis='x', rotation=45)
            axes[0, 2].grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, time_val in zip(bars, avg_times):
                axes[0, 2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(std_times) * 0.1,
                                f'{time_val:.1f}', ha='center', va='bottom', fontsize=9)

        # 4. Real sequence performance
        if real_results and analysis.get('real_sequence_analysis'):
            real_alg_names = []
            real_avg_identities = []
            real_std_identities = []

            for alg_name in algorithms:
                if alg_name in analysis['real_sequence_analysis']:
                    real_alg_names.append(alg_name)
                    real_avg_identities.append(analysis['real_sequence_analysis'][alg_name]['avg_identity'])
                    real_std_identities.append(analysis['real_sequence_analysis'][alg_name]['std_identity'])

            bars = axes[1, 0].bar(real_alg_names, real_avg_identities, yerr=real_std_identities,
                                  capsize=5, alpha=0.8, color=colors[:len(real_alg_names)])
            axes[1, 0].set_ylabel('Average identity (%)')
            axes[1, 0].set_title('Real sequence identity comparison')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)

        # 5. Score vs Identity scatter for real data
        if real_results:
            for i, alg_name in enumerate(algorithms):
                scores = []
                identities = []

                for comparison in real_results:
                    alg_result = comparison['algorithm_results'].get(alg_name)
                    if alg_result:
                        scores.append(alg_result['result'].score)
                        identities.append(alg_result['stats']['identity_percent'])

                if scores and identities:
                    axes[1, 1].scatter(scores, identities, alpha=0.7,
                                       label=alg_name, color=colors[i], s=40)

            axes[1, 1].set_xlabel('Alignment score')
            axes[1, 1].set_ylabel('Identity (%)')
            axes[1, 1].set_title('Score vs Identity relationship')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        # 6. Gap penalty comparison (test cases with gaps)
        if test_results:
            gap_test_names = []
            nw_gaps = []
            sw_gaps = []
            gotoh_nw_gaps = []
            gotoh_sw_gaps = []

            for test_result in test_results:
                if any(keyword in test_result['test_case'].lower()
                       for keyword in ['插入', '删除', 'gap', '长插入']):
                    gap_test_names.append(test_result['test_case'])

                    for alg_name in algorithms:
                        alg_result = test_result['algorithm_results'].get(alg_name)
                        if alg_result:
                            gaps = alg_result['stats']['gaps']
                            if alg_name == 'Needleman-Wunsch':
                                nw_gaps.append(gaps)
                            elif alg_name == 'Smith-Waterman':
                                sw_gaps.append(gaps)
                            elif alg_name == 'Gotoh-NW':
                                gotoh_nw_gaps.append(gaps)
                            elif alg_name == 'Gotoh-SW':
                                gotoh_sw_gaps.append(gaps)

            if gap_test_names:
                x = np.arange(len(gap_test_names))
                width = 0.2

                if len(nw_gaps) == len(gap_test_names):
                    axes[1, 2].bar(x - width * 1.5, nw_gaps, width, label='NW', alpha=0.8, color=colors[0])
                if len(sw_gaps) == len(gap_test_names):
                    axes[1, 2].bar(x - width * 0.5, sw_gaps, width, label='SW', alpha=0.8, color=colors[1])
                if len(gotoh_nw_gaps) == len(gap_test_names):
                    axes[1, 2].bar(x + width * 0.5, gotoh_nw_gaps, width, label='Gotoh-NW', alpha=0.8, color=colors[2])
                if len(gotoh_sw_gaps) == len(gap_test_names):
                    axes[1, 2].bar(x + width * 1.5, gotoh_sw_gaps, width, label='Gotoh-SW', alpha=0.8, color=colors[3])

                axes[1, 2].set_xlabel('Gap-containing test cases')
                axes[1, 2].set_ylabel('Number of gaps')
                axes[1, 2].set_title('Gap handling comparison')
                axes[1, 2].set_xticks(x)
                axes[1, 2].set_xticklabels(gap_test_names, rotation=45, ha='right')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)

        # 7. Algorithm complexity visualization (theoretical)
        complexity_data = {
            'Needleman-Wunsch': {'time': 'O(mn)', 'space': 'O(mn)', 'gap_model': 'Linear'},
            'Smith-Waterman': {'time': 'O(mn)', 'space': 'O(mn)', 'gap_model': 'Linear'},
            'Gotoh-NW': {'time': 'O(mn)', 'space': 'O(mn)', 'gap_model': 'Affine'},
            'Gotoh-SW': {'time': 'O(mn)', 'space': 'O(mn)', 'gap_model': 'Affine'}
        }

        axes[2, 0].text(0.5, 0.5, 'Algorithm Complexity\n\nAll algorithms: O(mn) time, O(mn) space\n\n'
                                  'Linear Gap: constant penalty per gap\n'
                                  'Affine Gap: open + extend penalties',
                        ha='center', va='center', fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        axes[2, 0].set_xlim(0, 1)
        axes[2, 0].set_ylim(0, 1)
        axes[2, 0].set_title('Algorithm complexity comparison')
        axes[2, 0].axis('off')

        # 8. Variant discrimination (if real data has variants)
        if real_results:
            variant_performance = {}

            for comparison in real_results:
                is_same_variant = comparison['same_variant']

                for alg_name in algorithms:
                    alg_result = comparison['algorithm_results'].get(alg_name)
                    if alg_result:
                        identity = alg_result['stats']['identity_percent']

                        if alg_name not in variant_performance:
                            variant_performance[alg_name] = {'same': [], 'different': []}

                        if is_same_variant:
                            variant_performance[alg_name]['same'].append(identity)
                        else:
                            variant_performance[alg_name]['different'].append(identity)

            # Box plot for variant discrimination
            same_data = []
            diff_data = []
            labels = []

            for alg_name in algorithms:
                if alg_name in variant_performance:
                    same_vals = variant_performance[alg_name]['same']
                    diff_vals = variant_performance[alg_name]['different']

                    if same_vals:
                        same_data.append(same_vals)
                        labels.append(f'{alg_name}\n(same variant)')
                    if diff_vals:
                        diff_data.append(diff_vals)
                        labels.append(f'{alg_name}\n(different variant)')

            if same_data or diff_data:
                all_data = same_data + diff_data
                all_labels = [f'{alg}\n(同变异株)' for alg in algorithms if
                              alg in variant_performance and variant_performance[alg]['same']]
                all_labels += [f'{alg}\n(不同变异株)' for alg in algorithms if
                               alg in variant_performance and variant_performance[alg]['different']]

                if all_data:
                    bp = axes[2, 1].boxplot(all_data[:8], labels=all_labels[:8])  # Limit for readability
                    axes[2, 1].set_ylabel('Identity (%)')
                    axes[2, 1].set_title('Variant discrimination ability')
                    axes[2, 1].tick_params(axis='x', rotation=45)
                    axes[2, 1].grid(True, alpha=0.3)

        # 9. Summary recommendations
        recommendations = self._generate_algorithm_recommendations(analysis)

        axes[2, 2].text(0.05, 0.95, recommendations, ha='left', va='top', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                        transform=axes[2, 2].transAxes, wrap=True)
        axes[2, 2].set_title('Algorithm selection recommendations')
        axes[2, 2].axis('off')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/comprehensive_algorithm_comparison.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_algorithm_recommendations(self, analysis: Dict) -> str:
        """Generate algorithm selection recommendations."""
        recommendations = "Algorithm selection recommendations:\n\n"

        recommendations += "🌐 Needleman-Wunsch:\n"
        recommendations += "• Global alignment; sequences of similar length\n"
        recommendations += "• Useful for evolutionary analyses and phylogenetics\n"
        recommendations += "• Good for overall sequence similarity scoring\n\n"

        recommendations += "🎯 Smith-Waterman:\n"
        recommendations += "• Local alignment; find conserved regions\n"
        recommendations += "• Useful for database searches\n"
        recommendations += "• Works well when sequence lengths differ greatly\n\n"

        recommendations += "🧬 Gotoh-NW:\n"
        recommendations += "• Global alignment with affine gap handling\n"
        recommendations += "• Preferred for protein alignments when gaps are realistic\n"
        recommendations += "• Better modeling of indel events\n\n"

        recommendations += "🔬 Gotoh-SW:\n"
        recommendations += "• Local alignment with affine gap handling\n"
        recommendations += "• Good for identifying functional domains\n"
        recommendations += "• Recommended for SARS-CoV-2 RBD-focused analyses\n\n"

        recommendations += "💡 SARS-CoV-2 guidance:\n"
        recommendations += "• Whole-genome variant comparisons: Gotoh-NW\n"
        recommendations += "• Conserved domain analysis: Gotoh-SW\n"
        recommendations += "• Fast screening: SW or Gotoh-SW"

        return recommendations

    def generate_comprehensive_report(self, test_results: List[Dict],
                                      real_results: List[Dict],
                                      analysis: Dict):
        """生成综合比较报告."""
        print("Generating comprehensive comparison report...")

        report_file = f"{self.output_dir}/comprehensive_comparison_report.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Comprehensive report for sequence alignment algorithm comparison\n")
            f.write("=" * 70 + "\n\n")
            f.write("Algorithms compared: Needleman-Wunsch vs Smith-Waterman vs Gotoh-NW vs Gotoh-SW\n\n")

            # Algorithm overview
            f.write("1. Algorithm overview\n")
            f.write("-" * 15 + "\n\n")

            f.write("Needleman-Wunsch (1970):\n")
            f.write("• Global alignment algorithm\n")
            f.write("• Linear gap penalty: w(k) = k × gap_penalty\n")
            f.write("• Suitable for homologous sequences of similar length\n")
            f.write("• Time complexity: O(mn), Space complexity: O(mn)\n\n")

            f.write("Smith-Waterman (1981):\n")
            f.write("• Local alignment algorithm\n")
            f.write("• Linear gap penalty with score reset to zero\n")
            f.write("• Suitable for finding local similarity regions\n")
            f.write("• Time complexity: O(mn), Space complexity: O(mn)\n\n")

            f.write("Gotoh-NW (1982):\n")
            f.write("• Global alignment with affine gap penalty\n")
            f.write("• Affine gap penalty: w(k) = gap_open + (k-1) × gap_extend\n")
            f.write("• More realistic gap modeling for indels\n")
            f.write("• Time complexity: O(mn), Space complexity: O(mn)\n\n")

            f.write("Gotoh-SW:\n")
            f.write("• Local alignment with affine gap penalty\n")
            f.write("• Combines local alignment and improved gap handling\n")
            f.write("• Suitable for complex local similarity analyses\n")
            f.write("• Time complexity: O(mn), Space complexity: O(mn)\n\n")

            # Test case results
            f.write("2. Test case results\n")
            f.write("-" * 18 + "\n\n")

            for i, test_result in enumerate(test_results[:8], 1):  # Limit for readability
                f.write(f"Test case {i}: {test_result['test_case']}\n")
                f.write(f"Description: {test_result['description']}\n")
                f.write(f"Sequence 1: {test_result['seq1']}\n")
                f.write(f"Sequence 2: {test_result['seq2']}\n")
                f.write(f"Expected: {test_result['expected']}\n\n")

                f.write("Results:\n")
                for alg_name in self.algorithms.keys():
                    alg_result = test_result['algorithm_results'].get(alg_name)
                    if alg_result:
                        result = alg_result['result']
                        stats = alg_result['stats']
                        time_ms = alg_result['execution_time'] * 1000

                        f.write(f"  {alg_name:15}: ")
                        f.write(f"Score={result.score:6.1f}, ")
                        f.write(f"Identity={stats['identity_percent']:5.1f}%, ")
                        f.write(f"Gap={stats['gaps']:2d}, ")
                        f.write(f"Time={time_ms:6.2f}ms\n")
                    else:
                        f.write(f"  {alg_name:15}: 执行失败\n")

                f.write("\n" + "-" * 60 + "\n\n")

            # Statistical analysis
            f.write("3. Statistical analysis\n")
            f.write("-" * 12 + "\n\n")

            if analysis.get('test_case_analysis'):
                f.write("Test case statistics (mean ± stddev):\n\n")
                for alg_name, stats in analysis['test_case_analysis'].items():
                    f.write(f"{alg_name}:\n")
                    f.write(f"  Average score: {stats['avg_score']:8.2f} ± {stats['std_score']:.2f}\n")
                    f.write(f"  Average identity: {stats['avg_identity']:6.2f}% ± {stats['std_identity']:.2f}%\n")
                    f.write(f"  Average execution time: {stats['avg_time_ms']:6.2f} ± {stats['std_time_ms']:.2f} ms\n\n")

            if analysis.get('real_sequence_analysis'):
                f.write("Real sequence statistics (mean ± stddev):\n\n")
                for alg_name, stats in analysis['real_sequence_analysis'].items():
                    f.write(f"{alg_name}:\n")
                    f.write(f"  Average score: {stats['avg_score']:8.2f} ± {stats['std_score']:.2f}\n")
                    f.write(f"  Average identity: {stats['avg_identity']:6.2f}% ± {stats['std_identity']:.2f}%\n")
                    f.write(f"  Average execution time: {stats['avg_time_ms']:6.2f} ± {stats['std_time_ms']:.2f} ms\n\n")

            # Real sequence comparison summary
            if real_results:
                f.write("4. Real sequence comparison summary\n")
                f.write("-" * 22 + "\n\n")
                f.write(f"Total comparisons: {len(real_results)}\n")

                same_variant_count = sum(1 for r in real_results if r['same_variant'])
                f.write(f"Comparisons within same variant: {same_variant_count}\n")
                f.write(f"Comparisons across different variants: {len(real_results) - same_variant_count}\n\n")

                # Variant distribution
                variants = {}
                for result in real_results:
                    v1 = result['seq1']['variant']
                    v2 = result['seq2']['variant']
                    variants[v1] = variants.get(v1, 0) + 1
                    variants[v2] = variants.get(v2, 0) + 1

                f.write("Variant distribution:\n")
                for variant, count in variants.items():
                    f.write(f"  {variant}: {count} occurrences\n")
                f.write("\n")

            # Recommendations
            f.write("5. Algorithm selection recommendations\n")
            f.write("-" * 18 + "\n\n")

            recommendations = self._generate_algorithm_recommendations(analysis)
            f.write(recommendations)
            f.write("\n\n")

            # Parameter recommendations
            f.write("6. Parameter recommendations\n")
            f.write("-" * 12 + "\n\n")
            f.write("Linear gap penalty parameters:\n")
            f.write(f"  Match score: {self.linear_params.match}\n")
            f.write(f"  Mismatch score: {self.linear_params.mismatch}\n")
            f.write(f"  Gap penalty: {self.linear_params.gap}\n\n")

            f.write("Affine gap penalty parameters:\n")
            f.write(f"  Match score: {self.affine_params.match}\n")
            f.write(f"  Mismatch score: {self.affine_params.mismatch}\n")
            f.write(f"  Gap open penalty: {self.affine_params.gap_open}\n")
            f.write(f"  Gap extend penalty: {self.affine_params.gap_extend}\n\n")

            f.write("For SARS-CoV-2 sequence analysis, recommendations:\n")
            f.write("• Nucleotide sequences: match=2, mismatch=-1\n")
            f.write("• Protein sequences: use BLOSUM62 matrix\n")
            f.write("• Adjust gap penalties based on biological relevance\n")
            f.write("• Affine gap models better reflect real indel events\n")

            # Files generated
            f.write("\n7. Output files\n")
            f.write("-" * 12 + "\n")
            f.write("• comprehensive_algorithm_comparison.png - comprehensive comparison figure\n")
            f.write("• comprehensive_comparison_report.txt - detailed report\n")

        print(f"Comprehensive report saved to: {report_file}")

    def run_complete_comparison(self):
        """运行完整的算法比较分析."""
        print("Starting comprehensive sequence alignment algorithm comparison")
        print("=" * 60)
        print(f"Algorithms compared: {', '.join(self.algorithms.keys())}")
        print(
            f"Linear gap params: match={self.linear_params.match}, mismatch={self.linear_params.mismatch}, gap={self.linear_params.gap}")
        print(
            f"Affine gap params: match={self.affine_params.match}, mismatch={self.affine_params.mismatch}, gap_open={self.affine_params.gap_open}, gap_extend={self.affine_params.gap_extend}")

        # 1. Run test cases
        test_results = self.run_test_cases()

        # 2. Run real sequence comparison
        sequence_files = [
            "spike_sequences/*.fasta",
            "reference_sequences/*.fasta",
            "recent_sequences/*.fasta"
        ]
        real_results = self.run_real_sequence_comparison(sequence_files)
        # Save real results for later analysis
        '''
        comparisons_to_save = []
        for comp in real_results:
            filtered_comp = comp.copy()
            filtered_alg_results = {}
            for alg_name, data in comp['algorithm_results'].items():
                #Dict without 'result' key to save space
                filtered_alg_results[alg_name] = {k: v for k, v in data.items() if k != 'result'}
            filtered_comp['algorithm_results'] = filtered_alg_results
            comparisons_to_save.append(filtered_comp)
        with open(f"{self.output_dir}/real_sequence_comparison_results.json", 'w', encoding='utf-8') as f:
            json.dump(comparisons_to_save, f, ensure_ascii=False, indent=4)
        print(f"Real sequence comparison results saved to: {self.output_dir}/real_sequence_comparison_results.json")
'''
        # 3. Analyze performance
        analysis = self.analyze_algorithm_performance(test_results, real_results)

        # 4. Generate visualizations
        self.visualize_comparison_results(test_results, real_results, analysis)

        # 5. Generate comprehensive report
        self.generate_comprehensive_report(test_results, real_results, analysis)

        print(f"\n✅ Comprehensive algorithm comparison complete!")
        print(f"📊 Results saved to: {self.output_dir}/")
        print(f"📈 Figure: comprehensive_algorithm_comparison.png")
        print(f"📋 Report: comprehensive_comparison_report.txt")

        # Print summary
        if analysis.get('test_case_analysis'):
            print("\n📊 Test case performance summary:")
            for alg_name, stats in analysis['test_case_analysis'].items():
                print(f"  {alg_name:12}: average identity {stats['avg_identity']:5.1f}%, "
                      f"average time {stats['avg_time_ms']:6.2f}ms")

        if analysis.get('real_sequence_analysis'):
            print("\n🧬 Real sequence performance summary:")
            for alg_name, stats in analysis['real_sequence_analysis'].items():
                print(f"  {alg_name:12}: average identity {stats['avg_identity']:5.1f}%, "
                      f"average time {stats['avg_time_ms']:6.2f}ms")


def main():
    comparator = AlignmentComparator()
    comparator.run_complete_comparison()


if __name__ == "__main__":
    main()