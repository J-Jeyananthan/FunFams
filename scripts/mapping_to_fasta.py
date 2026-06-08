from Bio import SeqIO

txt_file = "/SAN/orengolab/functional-families/janu/data/dup_removed_min3-funfams-4.3-c123-mapping.txt"
fasta_in = "/SAN/orengolab/functional-families/janu/data/funfams-4.3-c123.fasta"
fasta_out = "/SAN/orengolab/functional-families/janu/data/funfams-4.3-c123-no-dup-min3.fasta"

# 1. Load domain IDs into a set
domain_ids = set()
with open(txt_file) as f:
    for line in f:
        if not line.strip():
            continue
        domain_id = line.split("\t", 1)[0]
        domain_ids.add(domain_id)

print(f"Loaded {len(domain_ids):,} domain IDs")

# 2. Stream FASTA and write matches only
with open(fasta_out, "w") as out:
    for rec in SeqIO.parse(fasta_in, "fasta"):
        if rec.id in domain_ids:
            SeqIO.write(rec, out, "fasta")

print(f"Wrote filtered FASTA to {fasta_out}")
