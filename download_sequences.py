#!/usr/bin/env python3
"""
SARS-CoV-2 Spike Protein 序列下载脚本
"""

from Bio import Entrez, SeqIO
import os
import time
from typing import List, Dict

Entrez.email = ""

def download_by_accession(accession_list: List[str], filename: str, description: str) -> int:
    """根据登录号直接下载序列"""
    if not accession_list:
        return 0

    try:
        print(f"正在下载 {description} ({len(accession_list)} 条登录号)...")
        handle = Entrez.efetch(db="nucleotide", id=accession_list, rettype="fasta", retmode="text")
        sequences = handle.read()
        handle.close()

        with open(filename, 'w') as f:
            f.write(sequences)

        seq_count = sequences.count('>')
        print(f"成功下载 {seq_count} 条序列到 {filename}")
        return seq_count

    except Exception as e:
        print(f"下载 {description} 失败: {e}")
        return 0

def search_with_flexible_terms(base_terms: List[str], retmax: int = 50) -> List[str]:
    """使用多个搜索词进行灵活搜索"""
    all_ids = set()

    for term in base_terms:
        try:
            print(f"  搜索: {term}")
            handle = Entrez.esearch(db="nucleotide", term=term, retmax=retmax)
            search_results = Entrez.read(handle)
            handle.close()

            new_ids = set(search_results["IdList"])
            all_ids.update(new_ids)
            print(f"    找到 {len(new_ids)} 条序列")

            time.sleep(0.5)  # 避免请求过快
        except Exception as e:
            print(f"    搜索失败: {e}")

    return list(all_ids)

def download_known_reference_sequences():
    """下载已知的重要参考序列"""
    print("下载重要参考序列...")
    os.makedirs("reference_sequences", exist_ok=True)

    # 重要的参考序列登录号
    references = {
        "original_wuhan": [
            "NC_045512.2",     # 完整基因组
            "YP_009724390.1",  # Spike蛋白
            "QHD43416.1",      # 原始Spike
            "MN908947.3"       # 另一个完整基因组
        ],
        "alpha_variants": [
            "MW822591.1", "MW822592.1", "MW822593.1", "MW822594.1", "MW822595.1"
        ],
        "beta_variants": [
            "MW598020.1", "MW598021.1", "MW598022.1", "MW882132.1", "MW882133.1"
        ],
        "gamma_variants": [
            "MW979374.1", "MW979375.1", "MW979376.1", "MW979377.1", "MW979378.1"
        ],
        "delta_variants": [
            "OK091006.1", "OK091007.1", "OK091008.1", "OK011746.1", "OK011747.1"
        ],
        "omicron_variants": [
            "OM287563.1", "OM287564.1", "OM287565.1", "ON061016.1", "ON061017.1"
        ]
    }

    results = {}
    for variant, accessions in references.items():
        filename = f"reference_sequences/{variant}_known.fasta"
        count = download_by_accession(accessions, filename, variant)
        results[variant] = count
        time.sleep(1)

    return results

def download_by_flexible_search():
    """使用灵活搜索策略下载序列"""
    print("使用灵活搜索策略...")
    os.makedirs("spike_sequences", exist_ok=True)

    # 改进的搜索策略
    search_strategies = {
        "original_wuhan": [
            '"SARS-CoV-2"[Organism] AND "Wuhan"[All Fields] AND "spike"[All Fields]',
            '"severe acute respiratory syndrome coronavirus 2"[Organism] AND "spike glycoprotein"[All Fields] AND "Wuhan"[All Fields]',
            '"COVID-19"[All Fields] AND "spike protein"[All Fields] AND ("Wuhan" OR "original")',
        ],

        "alpha_b117": [
            '"SARS-CoV-2"[Organism] AND ("B.1.1.7" OR "Alpha variant" OR "501Y.V1")',
            '"COVID-19"[All Fields] AND ("B117" OR "B.1.1.7" OR "Alpha")',
            '"spike protein"[All Fields] AND ("B.1.1.7" OR "Alpha variant")',
        ],

        "beta_b1351": [
            '"SARS-CoV-2"[Organism] AND ("B.1.351" OR "Beta variant" OR "501Y.V2")',
            '"COVID-19"[All Fields] AND ("B1351" OR "B.1.351" OR "Beta")',
            '"spike protein"[All Fields] AND ("B.1.351" OR "Beta variant")',
        ],

        "gamma_p1": [
            '"SARS-CoV-2"[Organism] AND ("P.1" OR "Gamma variant" OR "501Y.V3")',
            '"COVID-19"[All Fields] AND ("P1" OR "P.1" OR "Gamma")',
            '"spike protein"[All Fields] AND ("P.1" OR "Gamma variant")',
        ],

        "delta_b16172": [
            '"SARS-CoV-2"[Organism] AND ("B.1.617.2" OR "Delta variant")',
            '"COVID-19"[All Fields] AND ("B16172" OR "B.1.617.2" OR "Delta")',
            '"spike protein"[All Fields] AND ("B.1.617.2" OR "Delta variant")',
        ],

        "omicron_b11529": [
            '"SARS-CoV-2"[Organism] AND ("B.1.1.529" OR "Omicron variant" OR "BA.1")',
            '"COVID-19"[All Fields] AND ("B11529" OR "B.1.1.529" OR "Omicron" OR "BA.1")',
            '"spike protein"[All Fields] AND ("Omicron" OR "B.1.1.529" OR "BA.1")',
        ]
    }

    results = {}

    for variant, search_terms in search_strategies.items():
        print(f"\n搜索 {variant} 变异株...")

        # 使用多个搜索词
        id_list = search_with_flexible_terms(search_terms, retmax=30)

        if id_list:
            filename = f"spike_sequences/{variant}_spike.fasta"
            try:
                handle = Entrez.efetch(db="nucleotide", id=id_list, rettype="fasta", retmode="text")
                sequences = handle.read()
                handle.close()

                with open(filename, 'w') as f:
                    f.write(sequences)

                seq_count = sequences.count('>')
                results[variant] = seq_count
                print(f"成功下载 {seq_count} 条序列")

            except Exception as e:
                print(f"下载失败: {e}")
                results[variant] = 0
        else:
            print("未找到序列")
            results[variant] = 0

        time.sleep(2)  # 避免请求过快

    return results

def download_recent_sequences():
    """下载最近的序列数据"""
    print("\n下载最近的序列数据...")

    # 按时间和地区搜索
    recent_searches = {
        "recent_global": '"SARS-CoV-2"[Organism] AND "spike"[All Fields] AND "2023"[Publication Date]',
        "recent_variants": '"COVID-19"[All Fields] AND "variant"[All Fields] AND "spike protein"[All Fields] AND "2023"[Publication Date]',
    }

    os.makedirs("recent_sequences", exist_ok=True)

    for name, query in recent_searches.items():
        try:
            handle = Entrez.esearch(db="nucleotide", term=query, retmax=20)
            search_results = Entrez.read(handle)
            handle.close()

            if search_results["IdList"]:
                filename = f"recent_sequences/{name}.fasta"
                handle = Entrez.efetch(db="nucleotide", id=search_results["IdList"],
                                       rettype="fasta", retmode="text")
                sequences = handle.read()
                handle.close()

                with open(filename, 'w') as f:
                    f.write(sequences)

                print(f"{name}: 下载了 {sequences.count('>')} 条序列")

            time.sleep(1)
        except Exception as e:
            print(f"{name} 下载失败: {e}")

def main():
    print("优化的SARS-CoV-2 Spike Protein 序列下载工具")
    print("=" * 60)

    # 测试连接
    try:
        handle = Entrez.einfo()
        Entrez.read(handle)
        handle.close()
        print("NCBI连接正常")
    except Exception as e:
        print(f"无法连接到NCBI: {e}")
        return

    print("\n=== 策略1: 下载已知重要参考序列 ===")
    ref_results = download_known_reference_sequences()

    print("\n=== 策略2: 使用灵活搜索策略 ===")
    search_results = download_by_flexible_search()

    print("\n=== 策略3: 下载最近序列 ===")
    download_recent_sequences()

    # 结果摘要
    print("\n" + "=" * 60)
    print("下载结果摘要:")
    print("\n已知参考序列:")
    total_ref = 0
    for variant, count in ref_results.items():
        print(f"  {variant:20}: {count:3d} 条序列")
        total_ref += count

    print("\n搜索获得序列:")
    total_search = 0
    for variant, count in search_results.items():
        print(f"  {variant:20}: {count:3d} 条序列")
        total_search += count

    print(f"\n总计: 参考序列 {total_ref} 条, 搜索序列 {total_search} 条")

    # 数据质量建议
    print("\n" + "=" * 60)
    print("📊 数据量建议:")
    print("• 每个变异株 5-20 条序列适合比对分析")
    print("• 总序列数 50-100 条为最佳范围")
    print("• 如果某个变异株序列过少，会在分析中补充已知序列")

if __name__ == "__main__":
    main()