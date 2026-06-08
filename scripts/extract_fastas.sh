#!/bin/bash
SEQKIT=/SAN/orengolab/functional-families/janu/seqkit/seqkit
TEST_FASTA=/SAN/orengolab/functional-families/janu/contrasted-ff/test.fasta
SPLITS_DIR=/SAN/orengolab/functional-families/janu/contrasted-ff/seq_sim_splits

for f in "$SPLITS_DIR"/test_le*.tsv; do
    name=$(basename "$f" .tsv)
    tail -n +2 "$f" | cut -f1 | "$SEQKIT" grep -f - "$TEST_FASTA" > "$SPLITS_DIR/${name}.fasta" && echo "Done: ${name}.fasta"
done
