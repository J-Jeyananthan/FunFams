"""
Extends calculate_ec_purity.py with EC split count analysis.

In addition to the existing purity metrics (how homogeneous are FunFams with respect to EC terms),
this script computes the inverse: for each EC4 term, how many FunFams does it appear in (split count).
This is a complementary metric to purity — purity measures cluster homogeneity, split count measures
how fragmented each EC class is across the clustering.

Per-EC-term split counts are saved to a CSV for plotting (e.g. CCDF, scatter plot comparing methods)
in a Jupyter notebook. Summary stats (median split count, % of EC terms contained in a single FunFam)
are printed to stdout alongside the existing purity stats.

New CLI argument: --output-csv
"""
import os
import csv
import statistics
import click


def read_ec_file(ec_file_path):
    """Return {uniprot_id: set(ec_terms)} from a CSV file with columns uniprot_id,ec_term."""
    ec_annotations = {}
    with open(ec_file_path, "r") as ec_file:
        for line in ec_file:
            uniprot_id, ec_term = line.strip().split(',')
            ec_annotations.setdefault(uniprot_id, set()).add(ec_term)
    return ec_annotations


def calculate_purities(ec_terms):
    """Return (ec4_purity, ec3_purity) for a list of EC terms from a single FunFam."""
    ec_counts = {ec: ec_terms.count(ec) for ec in set(ec_terms)}
    most_common_ec4 = max(ec_counts, key=ec_counts.get)
    most_common_ec3 = max(
        (ec.rsplit('.', 1)[0] for ec in ec_counts),
        key=lambda ec3: sum(count for ec, count in ec_counts.items() if ec.startswith(ec3))
    )
    ec4_purity = ec_counts[most_common_ec4] / len(ec_terms)
    ec3_purity = sum(
        count for ec, count in ec_counts.items() if ec.startswith(most_common_ec3)
    ) / len(ec_terms)
    return ec4_purity, ec3_purity

@click.command()
@click.option('--ec-file', required=True, type=click.Path(exists=True), help='Path to the EC annotation file.')
@click.option('--funfams-dir', required=True, type=click.Path(exists=True), help='Path to the folder containing FASTA alignments.')
@click.option('--output-csv', required=True, type=click.Path(), help='Path to save per-EC-term split count CSV.')
def calculate_ec_purity_and_split_proportion(ec_file, funfams_dir, output_csv):
    ec_annotations = read_ec_file(ec_file)

    ec4_purities = []
    ec3_purities = []

    alignments_with_ec = 0
    alignments_with_ec4_purity_over_80 = 0
    alignments_with_ec4_purity_over_90 = 0
    alignments_with_ec4_purity_100 = 0

    alignments_with_1_ec = 0
    alignments_with_2_ec = 0
    alignments_with_3_ec = 0
    alignments_with_4_or_more_ec = 0

    # Maps each EC4 term to the set of FunFams it appears in; used to compute split counts
    ec4_to_funfams = {}

    fasta_files = [f for f in os.listdir(funfams_dir) if f.endswith(".faa") or f.endswith(".aln")]
    total_funfams = len(fasta_files)

    for filename in fasta_files:
        alignment_path = os.path.join(funfams_dir, filename)
        ec_terms_in_alignment = []
        with open(alignment_path, "r") as alignment_file:
            for line in alignment_file:
                if line.startswith('>'):
                    uniprot_id = line.split('/')[0][1:]
                    ec_terms_in_alignment.extend(ec_annotations.get(uniprot_id, []))

        if not ec_terms_in_alignment:
            continue

        alignments_with_ec += 1
        ec4_purity, ec3_purity = calculate_purities(ec_terms_in_alignment)
        ec4_purities.append(ec4_purity)
        ec3_purities.append(ec3_purity)

        if ec4_purity > 0.8:
            alignments_with_ec4_purity_over_80 += 1
        if ec4_purity > 0.9:
            alignments_with_ec4_purity_over_90 += 1
        if ec4_purity == 1.0:
            alignments_with_ec4_purity_100 += 1

        unique_ec_count = len(set(ec_terms_in_alignment))
        if unique_ec_count == 1:
            alignments_with_1_ec += 1
        elif unique_ec_count == 2:
            alignments_with_2_ec += 1
        elif unique_ec_count == 3:
            alignments_with_3_ec += 1
        elif unique_ec_count >= 4:
            alignments_with_4_or_more_ec += 1

        # For each unique EC term in this FunFam, record that this FunFam contains it
        # Using set() so each FunFam is counted once per EC term even if multiple sequences share that term
        for ec_term in set(ec_terms_in_alignment):
            ec4_to_funfams.setdefault(ec_term, set()).add(filename)

    ff_percentage_with_ecs = (alignments_with_ec / total_funfams) * 100
    ec4_percentage_over_80 = (alignments_with_ec4_purity_over_80 / alignments_with_ec) * 100 if alignments_with_ec else 0
    ec4_percentage_over_90 = (alignments_with_ec4_purity_over_90 / alignments_with_ec) * 100 if alignments_with_ec else 0
    ec4_percentage_100 = (alignments_with_ec4_purity_100 / alignments_with_ec) * 100 if alignments_with_ec else 0
    avg_ec4_purity = sum(ec4_purities) / len(ec4_purities) if ec4_purities else 0
    avg_ec3_purity = sum(ec3_purities) / len(ec3_purities) if ec3_purities else 0

    percentage_with_1_ec = (alignments_with_1_ec / alignments_with_ec) * 100 if alignments_with_ec else 0
    percentage_with_2_ec = (alignments_with_2_ec / alignments_with_ec) * 100 if alignments_with_ec else 0
    percentage_with_3_ec = (alignments_with_3_ec / alignments_with_ec) * 100 if alignments_with_ec else 0
    percentage_with_4_or_more_ec = (alignments_with_4_or_more_ec / alignments_with_ec) * 100 if alignments_with_ec else 0

    print("FF%_withECs:", ff_percentage_with_ecs)
    print("ECpurity>80%:", ec4_percentage_over_80)
    print("ECpurity>90%:", ec4_percentage_over_90)
    print("ECpurity100%:", ec4_percentage_100)
    print("Average EC4 Purity:", avg_ec4_purity)
    print("Average EC3 Purity:", avg_ec3_purity)
    print("Percentage with 1 EC:", percentage_with_1_ec)
    print("Percentage with 2 EC:", percentage_with_2_ec)
    print("Percentage with 3 EC:", percentage_with_3_ec)
    print("Percentage with 4 or more EC:", percentage_with_4_or_more_ec)

    # split_count for each EC term = number of distinct FunFams it appears in
    split_counts = [len(funfams) for funfams in ec4_to_funfams.values()]
    ec_in_single_funfam = sum(1 for c in split_counts if c == 1)

    print("Total FunFams:", total_funfams)
    print("Total EC4 terms:", len(ec4_to_funfams))
    print("Median EC4 split count:", statistics.median(split_counts) if split_counts else 0)
    print("% EC4 terms in exactly 1 FunFam:", (ec_in_single_funfam / len(split_counts)) * 100 if split_counts else 0)

    # Save per-EC-term split counts to CSV for CCDF and scatter plot comparisons in Jupyter
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ec4_term', 'split_count'])
        for ec_term, funfams in ec4_to_funfams.items():
            writer.writerow([ec_term, len(funfams)])

if __name__ == '__main__':
    calculate_ec_purity_and_split_proportion()
