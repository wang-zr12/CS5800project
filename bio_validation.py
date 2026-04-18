#!/usr/bin/env python3
"""
biological_validation.py - Validate sequence alignment algorithms against real SARS-CoV-2 biology.

Uses the professional alignment implementations (including Gotoh variants) to verify
algorithms against known biological relationships, functional domains and variant
signatures.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from typing import Dict, List, Tuple
import json
from datetime import datetime
import time

# Import professional alignment implementations
from alignment import (
    needleman_wunsch, smith_waterman, gotoh_global, gotoh_local,
    ScoringParams, AffineScoringParams, AlignmentResult,
    calculate_alignment_stats, format_alignment_display
)


class BiologicalValidator:
    """Validator that checks alignment algorithms against known biological facts."""

    def __init__(self, output_dir: str = "biological_validation"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Scoring parameters
        self.linear_params = ScoringParams(match=2, mismatch=-1, gap=-2)
        self.affine_params = AffineScoringParams(match=2, mismatch=-1, gap_open=-10, gap_extend=-1)

        # Algorithm registry with professional implementations
        self.algorithms = {
            'Needleman-Wunsch': lambda s1, s2: needleman_wunsch(s1, s2, self.linear_params),
            'Smith-Waterman': lambda s1, s2: smith_waterman(s1, s2, self.linear_params),
            'Gotoh-NW': lambda s1, s2: gotoh_global(s1, s2, self.affine_params),
            'Gotoh-SW': lambda s1, s2: gotoh_local(s1, s2, self.affine_params)
        }

        # Known SARS-CoV-2 functional domains and critical mutations
        self.known_functional_domains = {
            "RBD": {  # Receptor Binding Domain
                "start": 319,  # 在Spike蛋白中的位置
                "end": 541,
                "function": "Binds to ACE2 receptor",
                "critical_mutations": ["N501Y", "E484K", "K417N", "L452R", "T478K"]
            },
            "NTD": {  # N-Terminal Domain
                "start": 14,
                "end": 305,
                "function": "N-terminal domain",
                "critical_mutations": ["L18F", "T20N", "P26S", "A27S", "D80A"]
            },
            "Furin_cleavage": {  # Furin切割位点
                "start": 682,
                "end": 685,
                "function": "Protease cleavage site (Furin)",
                "critical_mutations": ["P681H", "P681R"]
            }
        }

        # 已知变异株的特征突变
        self.variant_signatures = {
            "Alpha": ["N501Y", "A570D", "D614G", "P681H", "T716I", "S982A", "D1118H"],
            "Beta": ["K417N", "E484K", "N501Y", "D614G", "A701V"],
            "Gamma": ["K417T", "E484K", "N501Y", "D614G", "H655Y"],
            "Delta": ["L452R", "T478K", "D614G", "P681R", "D950N"],
            "Omicron": ["G339D", "S371L", "S373P", "S375F", "K417N", "N440K",
                        "G446S", "S477N", "T478K", "E484A", "Q493R", "G496S",
                        "Q498R", "N501Y", "Y505H", "T547K", "D614G", "H655Y"]
        }

    def load_reference_sequences(self) -> Dict[str, str]:
        """Load reference sequences from known files or fall back to built-in fragments."""
        references = {}

        # 尝试从多个来源加载参考序列
        reference_files = [
            "reference_sequences/NC_045512*",
            "reference_sequences/*original*",
            "reference_sequences/*wuhan*"
        ]

        import glob
        for pattern in reference_files:
            files = glob.glob(pattern + ".fasta")
            for file in files:
                try:
                    for record in SeqIO.parse(file, "fasta"):
                        if "spike" in record.description.lower() or "S protein" in record.description:
                            references["wuhan_spike"] = str(record.seq)
                        elif len(str(record.seq)) > 25000:  # 完整基因组
                            # 提取Spike蛋白序列 (基因组位置21563-25384)
                            genome_seq = str(record.seq)
                            spike_start = 21563 - 1  # 转换为0-based索引
                            spike_end = 25384
                            if spike_start < len(genome_seq):
                                spike_seq = genome_seq[spike_start:spike_end]
                                references["wuhan_spike_from_genome"] = spike_seq
                        break
                except:
                    continue

        # 如果没有找到文件，使用已知的参考序列片段
        if not references:
            # SARS-CoV-2 Spike蛋白RBD区域的已知序列
            references["wuhan_rbd"] = """
NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQI
APGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCY
FPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRD
IADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRA
GCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEI
LSRLLDNSYIEGTKTKSKGTSVCYEHCSCLSTHVSWLNRDWTDDKKPCNECVKPNFNTNEKLNCPDSPTRKCCQKCSTH
""".replace("\n", "").replace(" ", "")

        return references

    def extract_functional_domains(self, sequence: str) -> Dict[str, str]:
        """Extract functional domain subsequences from a spike sequence."""
        domains = {}

        for domain_name, domain_info in self.known_functional_domains.items():
            start = domain_info["start"] - 1  # 转换为0-based
            end = domain_info["end"]

            if start < len(sequence) and end <= len(sequence):
                domains[domain_name] = sequence[start:end]
            elif start < len(sequence):
                domains[domain_name] = sequence[start:]

        return domains

    def validate_algorithm_on_known_relationships(self, references: Dict[str, str]) -> Dict:
        """Validate alignment algorithms using known biological relationships."""
        print("Validating algorithms against known biological relationships...")

        validation_results = {
            "functional_domain_tests": [],
            "variant_discrimination_tests": [],
            "conservation_tests": []
        }

        if not references:
            print("Warning: No reference sequences found, skipping biological validation")
            return validation_results

        reference_seq = list(references.values())[0]

        # 1. Functional domain conservation tests
        print("\n1. Functional domain conservation tests")
        domains = self.extract_functional_domains(reference_seq)

        for domain_name, domain_seq in domains.items():
            if len(domain_seq) < 20:  # skip sequences that are too short
                continue

            # 创建带有轻微突变的序列
            mutated_seq = self.introduce_mutations(domain_seq, mutation_rate=0.05)

            print(f"  Testing domain: {domain_name} ({len(domain_seq)} aa)")

            # 使用所有四种算法比对
            domain_results = {}
            for alg_name, alg_func in self.algorithms.items():
                try:
                    start_time = time.time()
                    result = alg_func(domain_seq, mutated_seq)
                    execution_time = time.time() - start_time

                    stats = calculate_alignment_stats(result.seq1_aligned, result.seq2_aligned)

                    domain_results[alg_name] = {
                        'result': result,
                        'stats': stats,
                        'execution_time': execution_time
                    }

                    print(f"    {alg_name:12}: {stats['identity_percent']:5.1f}% identity")

                except Exception as e:
                    print(f"    {alg_name:12}: failed - {str(e)}")
                    domain_results[alg_name] = None

            validation_results["functional_domain_tests"].append({
                "domain": domain_name,
                "function": self.known_functional_domains[domain_name]["function"],
                "original_length": len(domain_seq),
                "algorithm_results": domain_results,
                "expected_high_conservation": True
            })
        # 2. Variant discrimination tests
        print("\n2. Variant discrimination tests")
        if len(reference_seq) > 200:  # 只在有足够长序列时进行测试

            # 自动检测当前加载的是不是核苷酸(DNA/RNA)
            # 如果前 50 个字符里全是 A, C, G, T, N，说明没经过翻译
            if all(c in 'ACGTNacgtn' for c in reference_seq[:50]):
                print("  [跳过] 变异识别测试需要氨基酸序列，但当前参考序列是核苷酸。请提供 Spike 蛋白的氨基酸 FASTA 文件。")
            else:
                # 猜测坐标偏移量：如果序列包含 RBD 片段特征，说明不是从头开始的
                # 注意：实际偏移量取决于你的截取位置，这里假设如果是 wuhan_rbd 则是从 319 开始 (索引差 318)
                offset = 318 if "NITNLCPFGEV" in reference_seq else 0

                if offset > 0:
                    print(f"  [提示] 检测到使用的是片段序列，自动应用坐标偏移: -{offset}")

                for variant_name, mutations in self.variant_signatures.items():
                    if len(mutations) > 10:  # 只测试主要变异株
                        continue

                    print(f"  Testing variant: {variant_name} ({len(mutations)} mutations)")

                    # 传入 offset
                    variant_seq = self.simulate_variant_sequence(reference_seq, mutations, offset=offset)

                    if variant_seq == reference_seq:
                        print(f"    [跳过] {variant_name} 变异序列生成失败，跳过比对。")
                        continue

                    variant_results = {}
                    for alg_name, alg_func in self.algorithms.items():
                        try:
                            start_time = time.time()
                            result = alg_func(reference_seq, variant_seq)
                            execution_time = time.time() - start_time

                            stats = calculate_alignment_stats(result.seq1_aligned, result.seq2_aligned)

                            variant_results[alg_name] = {
                                'result': result,
                                'stats': stats,
                                'execution_time': execution_time
                            }

                            print(f"    {alg_name:12}: {stats['identity_percent']:5.1f}% identity")

                        except Exception as e:
                            print(f"    {alg_name:12}: failed - {str(e)}")
                            variant_results[alg_name] = None

                    validation_results["variant_discrimination_tests"].append({
                        "variant": variant_name,
                        "mutation_count": len(mutations),
                        "algorithm_results": variant_results,
                        "expected_high_similarity": True
                    })
        return validation_results

    def introduce_mutations(self, sequence: str, mutation_rate: float = 0.05) -> str:
        """Introduce random substitutions into a sequence at the given rate."""
        import random

        seq_list = list(sequence)
        num_mutations = max(1, int(len(sequence) * mutation_rate))

        for _ in range(num_mutations):
            pos = random.randint(0, len(seq_list) - 1)
            original_aa = seq_list[pos]

            # 简单的氨基酸替换
            amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            new_aa = random.choice([aa for aa in amino_acids if aa != original_aa])
            seq_list[pos] = new_aa

        return ''.join(seq_list)

    def simulate_variant_sequence(self, reference_seq: str, mutations: List[str], offset: int = 0) -> str:
        """Simulate a variant sequence by applying a list of known amino-acid substitutions."""
        seq_list = list(reference_seq)
        applied_count = 0

        for mutation in mutations:
            if len(mutation) >= 3:
                try:
                    original_aa = mutation[0]
                    # 加入 offset 以适应截短的片段 (如 RBD)
                    position = int(mutation[1:-1]) - 1 - offset
                    new_aa = mutation[-1]

                    if 0 <= position < len(seq_list):
                        if seq_list[position] == original_aa:
                            seq_list[position] = new_aa
                            applied_count += 1
                        else:
                            # 打印为什么这个位置没有应用突变
                            print(
                                f"      [警告] 突变 {mutation} 失败: 期望位置有 '{original_aa}', 但实际是 '{seq_list[position]}'")
                    else:
                        print(f"      [警告] 突变 {mutation} 超出序列长度范围!")
                except Exception as e:
                    continue

        if applied_count == 0:
            print("      [错误] 没有任何突变被成功应用，生成的序列与原序列完全相同！")

        return ''.join(seq_list)
    def analyze_real_world_performance(self, sequence_files: List[str]) -> Dict:
        """Analyze performance on real-world sequence files and return summary results."""
        print("\nAnalyzing real-world performance...")

        # Collect sequences
        sequences = []
        for file_pattern in sequence_files:
            import glob
            for file in glob.glob(file_pattern):
                try:
                    for record in SeqIO.parse(file, "fasta"):
                        seq_str = str(record.seq)
                        if len(seq_str) > 2000:  # Skip sequences longer than 1500
                            print(f"Skipping sequence {record.id} (length {len(seq_str)})")
                            continue
                        if len(seq_str) > 100:  # Only process sufficiently long sequences
                            sequences.append({
                                "id": record.id,
                                "seq": seq_str,
                                "variant": self.identify_variant(record.id, record.description)
                            })
                        if len(sequences) >= 20:  # Limit sequence count
                            break
                except Exception as e:
                    print(f"Warning: Unable to read file {file}: {e}")
                    continue
                if len(sequences) >= 20:
                    break
            if len(sequences) >= 20:
                break

        if len(sequences) < 2:
            print("Warning: Not enough sequences found; skipping real-world performance analysis")
            return {}

        print(f"Analyzing {len(sequences)} real sequences...")

        performance_results = {
            "pairwise_comparisons": [],
            "execution_times": {alg: [] for alg in self.algorithms.keys()},
            "variant_clustering": {}
        }

        # 执行两两比对
        import time
        comparison_count = 0
        max_comparisons = 15  # 限制比较数量

        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                if comparison_count >= max_comparisons:
                    break

                seq1 = sequences[i]
                seq2 = sequences[j]

                print(f"Comparing {seq1['variant']} vs {seq2['variant']}")

                # 使用所有四种算法
                comparison_result = {
                    "seq1_variant": seq1["variant"],
                    "seq2_variant": seq2["variant"],
                    "same_variant": seq1["variant"] == seq2["variant"],
                    "algorithm_results": {}
                }

                for alg_name, alg_func in self.algorithms.items():
                    try:
                        start_time = time.time()
                        result = alg_func(seq1["seq"], seq2["seq"])
                        execution_time = time.time() - start_time

                        stats = calculate_alignment_stats(result.seq1_aligned, result.seq2_aligned)

                        comparison_result["algorithm_results"][alg_name] = {
                            'result': result,
                            'stats': stats,
                            'execution_time': execution_time
                        }

                        performance_results["execution_times"][alg_name].append(execution_time)

                    except Exception as e:
                        print(f"  {alg_name} failed: {str(e)}")
                        comparison_result["algorithm_results"][alg_name] = None

                performance_results["pairwise_comparisons"].append(comparison_result)
                comparison_count += 1

            if comparison_count >= max_comparisons:
                break

        return performance_results

    def identify_variant(self, seq_id: str, description: str) -> str:
        """Identify variant from sequence ID or description (heuristic)."""
        text = (seq_id + " " + description).lower()

        variant_keywords = {
            "alpha": ["alpha", "b.1.1.7", "b117"],
            "beta": ["beta", "b.1.351", "b1351"],
            "gamma": ["gamma", "p.1", "p1"],
            "delta": ["delta", "b.1.617.2", "b16172"],
            "omicron": ["omicron", "b.1.1.529", "ba.1", "ba.2", "ba.4", "ba.5"]
        }

        for variant, keywords in variant_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return variant

        if "wuhan" in text or "original" in text or "nc_045512" in text:
            return "original"

        return "unknown"

    def visualize_validation_results(self, validation_results: Dict, performance_results: Dict):
        """Visualize validation results and performance summaries."""
        print("Generating validation visualizations...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Biological validation results: comparison of 4 alignment algorithms', fontsize=16)

        algorithms = list(self.algorithms.keys())
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

        # 1. 功能域保守性测试
        if validation_results.get("functional_domain_tests"):
            domain_data = validation_results["functional_domain_tests"]
            domains = [d["domain"] for d in domain_data]

            # Prepare data for all algorithms
            alg_identities = {alg: [] for alg in algorithms}

            for domain in domain_data:
                for alg_name in algorithms:
                    alg_result = domain["algorithm_results"].get(alg_name)
                    if alg_result and alg_result['stats']:
                        alg_identities[alg_name].append(alg_result['stats']['identity_percent'])
                    else:
                        alg_identities[alg_name].append(0)

            x = np.arange(len(domains))
            width = 0.2

            for i, alg_name in enumerate(algorithms):
                if alg_identities[alg_name]:
                    axes[0, 0].bar(x + i * width, alg_identities[alg_name], width,
                                   label=alg_name, alpha=0.8, color=colors[i])

            axes[0, 0].set_xlabel('Domain')
            axes[0, 0].set_ylabel('Sequence identity (%)')
            axes[0, 0].set_title('Functional domain conservation')
            axes[0, 0].set_xticks(x + width * 1.5)
            axes[0, 0].set_xticklabels(domains, rotation=45)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # 2. 变异株辨别测试
        if validation_results.get("variant_discrimination_tests"):
            variant_data = validation_results["variant_discrimination_tests"]
            variants = [d["variant"] for d in variant_data]

            # Prepare data for all algorithms
            variant_identities = {alg: [] for alg in algorithms}

            for variant in variant_data:
                for alg_name in algorithms:
                    alg_result = variant["algorithm_results"].get(alg_name)
                    if alg_result and alg_result['stats']:
                        variant_identities[alg_name].append(alg_result['stats']['identity_percent'])
                    else:
                        variant_identities[alg_name].append(0)

            x = np.arange(len(variants))

            for i, alg_name in enumerate(algorithms):
                if variant_identities[alg_name]:
                    axes[0, 1].bar(x + i * width, variant_identities[alg_name], width,
                                   label=alg_name, alpha=0.8, color=colors[i])

            axes[0, 1].set_xlabel('Variant')
            axes[0, 1].set_ylabel('Identity to reference (%)')
            axes[0, 1].set_title('Variant recognition test')
            axes[0, 1].set_xticks(x + width * 1.5)
            axes[0, 1].set_xticklabels(variants, rotation=45)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # 3. 执行时间比较
        if performance_results.get("execution_times"):
            time_data = []
            time_labels = []

            for alg_name in algorithms:
                if alg_name in performance_results["execution_times"]:
                    times = performance_results["execution_times"][alg_name]
                    if times:
                        time_data.append([t * 1000 for t in times])  # Convert to ms
                        time_labels.append(alg_name)

            if time_data:
                axes[0, 2].boxplot(time_data, labels=time_labels)
                axes[0, 2].set_ylabel('Execution time (ms)')
                axes[0, 2].set_title('Execution time distribution')
                axes[0, 2].tick_params(axis='x', rotation=45)
                axes[0, 2].grid(True, alpha=0.3)

        # 4. 真实序列相似度比较
        if performance_results.get("pairwise_comparisons"):
            comparisons = performance_results["pairwise_comparisons"]

            # Create scatter plot for first two algorithms
            if len(algorithms) >= 2:
                alg1, alg2 = algorithms[0], algorithms[1]
                alg1_scores = []
                alg2_scores = []

                for comparison in comparisons:
                    result1 = comparison["algorithm_results"].get(alg1)
                    result2 = comparison["algorithm_results"].get(alg2)

                    if result1 and result2:
                        alg1_scores.append(result1['stats']['identity_percent'])
                        alg2_scores.append(result2['stats']['identity_percent'])

                if alg1_scores and alg2_scores:
                    axes[1, 0].scatter(alg1_scores, alg2_scores, alpha=0.7)
                    axes[1, 0].plot([0, 100], [0, 100], 'r--', alpha=0.5)
                    axes[1, 0].set_xlabel(f'{alg1} identity (%)')
                    axes[1, 0].set_ylabel(f'{alg2} identity (%)')
                    axes[1, 0].set_title('Algorithm identity comparison')
                    axes[1, 0].grid(True, alpha=0.3)

        # 5. 算法性能总结
        if performance_results.get("execution_times"):
            avg_times = []
            alg_names = []

            for alg_name in algorithms:
                if alg_name in performance_results["execution_times"]:
                    times = performance_results["execution_times"][alg_name]
                    if times:
                        avg_times.append(np.mean(times) * 1000)  # ms
                        alg_names.append(alg_name)

            if avg_times:
                bars = axes[1, 1].bar(alg_names, avg_times, alpha=0.8, color=colors[:len(alg_names)])
                axes[1, 1].set_ylabel('Average execution time (ms)')
                axes[1, 1].set_title('Algorithm performance comparison')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True, alpha=0.3)

                # Add value labels on bars
                for bar, time_val in zip(bars, avg_times):
                    axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(avg_times) * 0.01,
                                    f'{time_val:.1f}', ha='center', va='bottom')

        # 6. 建议和总结
        recommendations = self._generate_validation_recommendations(validation_results, performance_results)

        axes[1, 2].text(0.05, 0.95, recommendations, ha='left', va='top', fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                         transform=axes[1, 2].transAxes, wrap=True)
        axes[1, 2].set_title('Algorithm selection recommendations')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/biological_validation_results.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_validation_recommendations(self, validation_results: Dict, performance_results: Dict) -> str:
        """Generate algorithm recommendations based on biological validation results."""
        recommendations = "Recommendations based on SARS-CoV-2 validation:\n\n"

        # Analyze functional domain performance
        if validation_results.get("functional_domain_tests"):
            avg_performance = {}
            for alg_name in self.algorithms.keys():
                identities = []
                for test in validation_results["functional_domain_tests"]:
                    result = test["algorithm_results"].get(alg_name)
                    if result and result['stats']:
                        identities.append(result['stats']['identity_percent'])

                if identities:
                    avg_performance[alg_name] = np.mean(identities)

            if avg_performance:
                best_alg = max(avg_performance, key=avg_performance.get)
                recommendations += f"🧬 Best for functional domain analysis: {best_alg}\n"
                recommendations += f"Average conservation: {avg_performance[best_alg]:.1f}%\n\n"

        # Add specific recommendations
        recommendations += "🎯 Recommended uses:\n"
        recommendations += "• Gotoh-NW: whole-genome alignments\n"
        recommendations += "• Gotoh-SW: RBD / domain-focused analysis ⭐\n"
        recommendations += "• Needleman-Wunsch: evolutionary analyses\n"
        recommendations += "• Smith-Waterman: quick local screening\n\n"

        recommendations += "💡 SARS-CoV-2 notes:\n"
        recommendations += "• Affine gap penalties better reflect viral indel biology\n"
        recommendations += "• Local alignment helps with variant-focused analyses\n"
        recommendations += "• Gotoh variants are recommended for real-data scenarios"

        return recommendations

    def generate_validation_report(self, validation_results: Dict, performance_results: Dict):
        """Generate a textual biological validation report (saved to output dir)."""
        print("Generating biological validation report...")

        report_file = f"{self.output_dir}/biological_validation_report.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("SARS-CoV-2 sequence alignment biological validation report\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Validation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 1. Methods overview
            f.write("1. Methods overview\n")
            f.write("-" * 20 + "\n")
            f.write("This validation is based on known SARS-CoV-2 biological features:\n")
            f.write("• Functional domain conservation: critical domains should remain conserved\n")
            f.write("• Variant recognition: algorithms should discriminate different variants\n")
            f.write("• Real data performance: behavior on actual sequence datasets\n\n")

            # 2. 功能域验证结果
            if validation_results.get("functional_domain_tests"):
                f.write("2. Functional domain conservation\n")
                f.write("-" * 25 + "\n")

                domain_tests = validation_results["functional_domain_tests"]
                for test in domain_tests:
                    f.write(f"\nDomain: {test['domain']}\n")
                    f.write(f"Function: {test['function']}\n")
                    f.write(f"Length: {test['original_length']} aa\n")
                    # Note: the validation framework stores per-algorithm stats under algorithm_results
                    # The legacy keys 'nw_identity'/'sw_identity' are optional; try to read from stored stats.
                    nw_id = None
                    sw_id = None
                    try:
                        nw_entry = test['algorithm_results'].get('Needleman-Wunsch')
                        sw_entry = test['algorithm_results'].get('Smith-Waterman')
                        if nw_entry and nw_entry.get('stats'):
                            nw_id = nw_entry['stats'].get('identity_percent')
                        if sw_entry and sw_entry.get('stats'):
                            sw_id = sw_entry['stats'].get('identity_percent')
                    except Exception:
                        pass

                    f.write(f"Needleman-Wunsch identity: {nw_id if nw_id is not None else 'N/A'}\n")
                    f.write(f"Smith-Waterman identity: {sw_id if sw_id is not None else 'N/A'}\n")

                    if (nw_id is not None and sw_id is not None and nw_id > 90 and sw_id > 90):
                        f.write("PASS: Domain conservation is high\n")
                    else:
                        f.write("WARNING: Domain conservation may be reduced; investigate further\n")
                    f.write("-" * 40 + "\n")

            # 3. 变异株识别验证
            if validation_results.get("variant_discrimination_tests"):
                f.write("\n3. Variant discrimination validation\n")
                f.write("-" * 20 + "\n")

                variant_tests = validation_results["variant_discrimination_tests"]
                for test in variant_tests:
                    f.write(f"\nVariant: {test['variant']}\n")
                    f.write(f"Mutation count: {test['mutation_count']}\n")
                    # Try to fetch identities from stored algorithm_results
                    nw_id = None
                    sw_id = None
                    try:
                        nw_entry = test['algorithm_results'].get('Needleman-Wunsch')
                        sw_entry = test['algorithm_results'].get('Smith-Waterman')
                        if nw_entry and nw_entry.get('stats'):
                            nw_id = nw_entry['stats'].get('identity_percent')
                        if sw_entry and sw_entry.get('stats'):
                            sw_id = sw_entry['stats'].get('identity_percent')
                    except Exception:
                        pass

                    f.write(f"Needleman-Wunsch identity: {nw_id if nw_id is not None else 'N/A'}\n")
                    f.write(f"Smith-Waterman identity: {sw_id if sw_id is not None else 'N/A'}\n")

                    # Reasonable identity range heuristic for variants
                    if (nw_id is not None and sw_id is not None and 80 <= nw_id <= 98 and 80 <= sw_id <= 98):
                        f.write("PASS: Identities are within expected range\n")
                    else:
                        f.write("WARNING: Unexpected identity values; review alignment details\n")
                    f.write("-" * 40 + "\n")

            # 4. 真实数据性能分析
            if performance_results.get("pairwise_comparisons"):
                f.write("\n4. Real data performance analysis\n")
                f.write("-" * 22 + "\n")

                comparisons = performance_results["pairwise_comparisons"]

                # Attempt to extract identity statistics for NW and SW from per-comparison algorithm_results
                same_variant_nw = []
                diff_variant_nw = []
                same_variant_sw = []
                diff_variant_sw = []
                for c in comparisons:
                    try:
                        nw_entry = c['algorithm_results'].get('Needleman-Wunsch')
                        sw_entry = c['algorithm_results'].get('Smith-Waterman')
                        if nw_entry and nw_entry.get('stats'):
                            if c.get('same_variant'):
                                same_variant_nw.append(nw_entry['stats'].get('identity_percent'))
                            else:
                                diff_variant_nw.append(nw_entry['stats'].get('identity_percent'))
                        if sw_entry and sw_entry.get('stats'):
                            if c.get('same_variant'):
                                same_variant_sw.append(sw_entry['stats'].get('identity_percent'))
                            else:
                                diff_variant_sw.append(sw_entry['stats'].get('identity_percent'))
                    except Exception:
                        continue

                f.write(f"Total comparisons: {len(comparisons)}\n")
                f.write(f"Within-variant comparisons: {len(same_variant_nw)}\n")
                f.write(f"Across-variant comparisons: {len(diff_variant_nw)}\n\n")

                if same_variant_nw:
                    f.write("Within-variant identity statistics:\n")
                    f.write(f"  NW mean: {np.mean(same_variant_nw):.2f}% ± {np.std(same_variant_nw):.2f}%\n")
                    f.write(f"  SW mean: {np.mean(same_variant_sw):.2f}% ± {np.std(same_variant_sw):.2f}%\n")

                if diff_variant_nw:
                    f.write("\nAcross-variant identity statistics:\n")
                    f.write(f"  NW mean: {np.mean(diff_variant_nw):.2f}% ± {np.std(diff_variant_nw):.2f}%\n")
                    f.write(f"  SW mean: {np.mean(diff_variant_sw):.2f}% ± {np.std(diff_variant_sw):.2f}%\n")

                # Execution time statistics (try keys in execution_times dict)
                nw_times = performance_results.get('execution_times', {}).get('Needleman-Wunsch', [])
                sw_times = performance_results.get('execution_times', {}).get('Smith-Waterman', [])

                if nw_times:
                    f.write(f"\nExecution time statistics:\n")
                    f.write(f"  Needleman-Wunsch: {np.mean(nw_times) * 1000:.2f} ± {np.std(nw_times) * 1000:.2f} ms\n")
                if sw_times:
                    f.write(f"  Smith-Waterman: {np.mean(sw_times) * 1000:.2f} ± {np.std(sw_times) * 1000:.2f} ms\n")

            # 5. 结论和建议
            f.write("\n5. Conclusions and recommendations\n")
            f.write("-" * 15 + "\n")

            f.write("Conclusions from biological validation:\n\n")

            # Recommendation heuristics based on domain tests
            if validation_results.get("functional_domain_tests"):
                # compute average identities for NW/SW if available
                nw_vals = []
                sw_vals = []
                for t in validation_results["functional_domain_tests"]:
                    try:
                        nw_entry = t['algorithm_results'].get('Needleman-Wunsch')
                        sw_entry = t['algorithm_results'].get('Smith-Waterman')
                        if nw_entry and nw_entry.get('stats'):
                            nw_vals.append(nw_entry['stats'].get('identity_percent'))
                        if sw_entry and sw_entry.get('stats'):
                            sw_vals.append(sw_entry['stats'].get('identity_percent'))
                    except Exception:
                        continue

                domain_avg_nw = np.mean(nw_vals) if nw_vals else None
                domain_avg_sw = np.mean(sw_vals) if sw_vals else None

                if domain_avg_nw is not None and domain_avg_sw is not None:
                    if domain_avg_nw > domain_avg_sw:
                        f.write("• Functional domain analysis: Needleman-Wunsch performs better in conservation detection\n")
                    else:
                        f.write("• Functional domain analysis: Smith-Waterman performs better in conservation detection\n")

            # Execution efficiency heuristic
            exec_times = performance_results.get('execution_times', {})
            nw_times = exec_times.get('Needleman-Wunsch', [])
            sw_times = exec_times.get('Smith-Waterman', [])
            if nw_times and sw_times:
                avg_nw_time = np.mean(nw_times)
                avg_sw_time = np.mean(sw_times)
                if avg_nw_time < avg_sw_time:
                    f.write("• Execution: Needleman-Wunsch is faster on average\n")
                else:
                    f.write("• Execution: Smith-Waterman is faster on average\n")

            f.write("\nSARS-CoV-2 application guidance:\n")
            f.write("• Whole-genome comparisons and evolutionary analysis: Needleman-Wunsch\n")
            f.write("• Functional domain and critical site analysis: Smith-Waterman\n")
            f.write("• Variant quick-screening: combine both approaches\n")
            f.write("• Large-scale database searches: Smith-Waterman recommended\n")

        print(f"Biological validation report saved to: {report_file}")

    def run_biological_validation(self):
        """Run full biological validation pipeline (load refs, validate, analyze, visualize, report)."""
        print("Starting biological validation based on SARS-CoV-2 data")
        print("=" * 50)

        # 1. 加载参考序列
        references = self.load_reference_sequences()

        def save_without_matrix(data):
            """递归遍历数据结构，移除 'result' 键以及跳过/转换 numpy 数组"""
            if isinstance(data, dict):
                cleaned_dict = {}
                for k, v in data.items():
                    if k == 'result':
                        continue  # 跳过包含对齐结果对象的键
                    if isinstance(v, np.ndarray):
                        continue  # 跳过 numpy 数组（如果想保留可以改为 v.tolist()）
                    cleaned_dict[k] = save_without_matrix(v)
                return cleaned_dict
            elif isinstance(data, list):
                # 如果是列表，递归处理列表中的每一个元素
                return [save_without_matrix(item) for item in data]
            else:
                # 基本数据类型（int, float, str, bool, None）直接返回
                return data
        # 2. 基于已知生物学关系验证
        validation_results = self.validate_algorithm_on_known_relationships(references)
        to_save_1=save_without_matrix(validation_results)
        # Save intermediate validation results for debugging
        with open(f"{self.output_dir}/validation_results.json", 'w', encoding='utf-8') as f:
            json.dump(to_save_1, f, indent=2)

        # 3. 真实序列性能分析
        sequence_files = [
            "spike_sequences/*.fasta",
            "reference_sequences/*.fasta",
            "recent_sequences/*.fasta"
        ]
        performance_results = self.analyze_real_world_performance(sequence_files)
        to_save_2 = save_without_matrix(performance_results)
        with open(f"{self.output_dir}/performance_results.json", 'w', encoding='utf-8') as f:
            json.dump(to_save_2, f, indent=2)

        self.visualize_validation_results(validation_results, performance_results)
        self.generate_validation_report(validation_results, performance_results)

        print(f"\n✅ Biological validation complete!")
        print(f"📊 Results saved to: {self.output_dir}/")
        print(f"📈 Figure: biological_validation_results.png")
        print(f"📋 Report: biological_validation_report.txt")


def main():
    validator = BiologicalValidator()
    validator.run_biological_validation()


if __name__ == "__main__":
    main()