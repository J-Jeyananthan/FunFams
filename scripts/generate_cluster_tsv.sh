#!/bin/bash
# Generates a MMseqs2-style cluster TSV (rep_id\tmember_id) from a directory of
# subdirectories each containing .faa files, where the first sequence in each
# .faa file is the cluster rep.
#
# Usage: bash generate_cluster_tsv.sh <clusters_dir> <output_path>
# Example: bash generate_cluster_tsv.sh /path/to/clusters_dir /path/to/output.tsv

clusters_dir="$1"
output_path="$2"

for f in "$clusters_dir"/*/*.faa; do
    rep=$(grep '^>' "$f" | head -1 | sed 's/^>//' | cut -d' ' -f1)
    grep '^>' "$f" | sed 's/^>//' | cut -d' ' -f1 | awk -v r="$rep" '{print r"\t"$0}'
done > "$output_path"
