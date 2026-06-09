# Density-based clustering and contrastive fine-tuning of protein language model embeddings for scalable functional family generation

**Janusan Jeyananthan** — MSci Bioinformatics, UCL (2025–26)  
Supervisors: Prof. Christine Orengo, David Miller, Dr. Nicola Bordin

---

## Project overview

This repository contains the code for my MSci research project, which investigates two complementary approaches to generating CATH Functional Families (FunFams):

1. **HDBSCAN clustering** of protein language model (pLM) embeddings as a scalable alternative to existing hierarchical methods (eMMA, MARC). Applied to ProstT5 and ESM2 embeddings of S90 cluster representatives across three CATH superfamilies.

2. **Supervised contrastive fine-tuning** using a proxy anchor loss to reshape pLM embedding spaces prior to clustering. Evaluated using FunFam labels from MARC as supervision signal.

Benchmarks span three CATH superfamilies: HUPs (3.40.50.620), aldolases (3.20.20.70), and the ThDP-binding fold (3.40.50.970). HDBSCAN reduces clustering runtime by over three orders of magnitude relative to eMMA whilst achieving comparable or higher EC4 purity.

---

## Repository structure

```
.
├── src/                        # Contrastive learning framework
│   ├── train.py                # Hydra CLI entry point
│   ├── model.py                # ProjectionHead (MLP) + FunfamSupConModel (LightningModule)
│   ├── data.py                 # LMDB-backed dataset and DataModule
│   ├── losses.py               # SupConLoss and ProxyAnchorLoss implementations
│   ├── callbacks.py            # k-NN evaluation callback (FAISS)
│   ├── faiss_utils.py          # FAISS index build/search utilities
│   └── utils.py                # Seed, FASTA parsing, label loading helpers
├── configs/
│   ├── train.yaml              # Main Hydra config (data paths, model, training)
│   └── experiment/
│       ├── proxy_anchor.yaml   # Proxy-anchor loss override (used in paper)
│       └── supcon.yaml         # Supervised contrastive loss override
├── scripts/
│   ├── embed_prostt5.py        # Generate ProstT5 embeddings → LMDB
│   ├── embed-esm2.py           # Generate ESM2 embeddings → LMDB
│   ├── extract_raw_prostt5.py  # Export ProstT5 embeddings from LMDB → .pt
│   ├── lmdb_to_pt.py           # Export ESM2 embeddings from LMDB → .pt
│   ├── hdbscan_benchmark.py    # HDBSCAN clustering + EC purity evaluation
│   ├── calculate_ec_purity.py  # EC4/EC3 purity metrics (called by hdbscan_benchmark.py)
│   ├── calculate_ec_purity_and_split_proportion.py  # Purity + EC term split count analysis (Figure 6)
│   ├── count_funfam_sizes.py   # FunFam size distribution statistics (Figure 5, Table 1)
│   ├── datasplit.py            # Train/validation split for contrastive training
│   ├── extract_embeddings.py   # Project embeddings through trained model → .pt
│   └── *.qsub                  # HPC job scripts (SGE scheduler, UCL cluster)
└── pyproject.toml
```

---

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management (Python 3.9+).

```bash
git clone https://github.com/J-Jeyananthan/FunFams.git
cd FunFams

# Create virtual environment and install core dependencies
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Core dependencies (training + clustering + evaluation)
uv pip install -e .

# Add embedding generation dependencies (transformers for ProstT5/ESM2)
uv pip install -e ".[embeddings]"

# Optional: FAISS for k-NN evaluation
uv pip install -e ".[faiss]"
```

Alternatively, with standard pip:
```bash
pip install -e ".[embeddings]"
```

---

## Data

All sequence data is from **CATHv4.3**. Due to their size, raw sequences, embeddings, and cluster files are stored on the UCL HPC filesystem and are not included in this repository.

| Dataset | Location (UCL HPC) | Description |
|---|---|---|
| Training sequences (8.6M) | `/SAN/orengolab/functional-families/janu/contrasted-ff/` | Seed FunFams from MARC, deduplicated |
| ProstT5 LMDB (training) | `/SAN/orengolab/functional-families/janu/contrasted-ff/funfams-4.3-c123.lmdb` | Float16, 1024-dim |
| ESM2 LMDB (training) | `/SAN/orengolab/functional-families/janu/contrasted-ff/funfams-4.3-c123-esm2.lmdb` | Float16, 1280-dim |
| FunFam label mapping | `/SAN/orengolab/functional-families/janu/data/duplicates_removed-funfams-4.3-c123-mapping.txt` | domain_id → FunFam label |
| S90 cluster TSVs | Per-superfamily on HPC | MMseqs2 output (rep → members) |
| EC annotations | Per-superfamily on HPC | Retrieved via UniProt ID Mapping |

S90 clustering was performed with MMseqs2 (`easy-cluster`, sensitivity 7.5, `--min-seq-id 0.9`, coverage mode 4, coverage threshold 0.95).

EC annotations were retrieved from the [UniProt ID Mapping Service](https://www.uniprot.org/help/id_mapping).

---

## Reproducing results

The full pipeline has two branches: HDBSCAN benchmarks (Figures 3–6, Table 1) and contrastive learning evaluation (Figure 7, Table 2). Both require pre-generated embeddings as input.

### Step 1 — Generate embeddings

**ProstT5** (used for HDBSCAN baseline and contrastive training):
```bash
python scripts/embed_prostt5.py \
    --input sequences.fasta \
    --output embeddings.lmdb \
    --per_protein 1 --half 1 --is_3Di 0
```

**ESM2-650M** (used for HDBSCAN baseline and contrastive training):
```bash
python scripts/embed-esm2.py \
    --input sequences.fasta \
    --output embeddings.lmdb \
    --model facebook/esm2_t33_650M_UR50D \
    --per_protein 1 --half 1
```

On the UCL HPC cluster, embedding jobs were submitted via the `.qsub` scripts in `scripts/` (e.g. `embed_3.20.20.70_prostt5.qsub`).

### Step 2 — Export embeddings to .pt format (for HDBSCAN input)

```bash
# ProstT5 baseline embeddings
python scripts/extract_raw_prostt5.py \
    --lmdb_path embeddings.lmdb \
    --fasta s90_reps.fasta \
    --output_dir output/

# ESM2 baseline embeddings
python scripts/lmdb_to_pt.py \
    --lmdb_path embeddings.lmdb \
    --output output/embeddings_esm2.pt
```

### Step 3 — HDBSCAN clustering and EC purity (Figures 3–6, Table 1)

```bash
python scripts/hdbscan_benchmark.py \
    --pt_file output/E_starting_prostt5.pt \
    --cluster_tsv s90_cluster.tsv \
    --ec_file ec_annotations.csv \
    --output_dir output/hdbscan_results/
```

This runs HDBSCAN (min_cluster_size=2, Euclidean distance on L2-normalised embeddings), expands clusters from S90 representatives to full members, writes per-FunFam `.faa` files, and prints EC4/EC3 purity metrics.

For the EC term split count analysis (Figure 6):
```bash
python scripts/calculate_ec_purity_and_split_proportion.py \
    --funfams-dir output/hdbscan_results/ \
    --ec-file ec_annotations.csv \
    --output-csv output/split_counts.csv
```

For FunFam size distribution statistics (Figure 5, Table 1):
```bash
python scripts/count_funfam_sizes.py output/hdbscan_results/
```

### Step 4 — Contrastive training (Figure 7, Table 2)

Split the training data:
```bash
python scripts/datasplit.py \
    --fasta_file train_deduped.fasta \
    --mapping_file funfam_labels.txt \
    --output_dir splits/ \
    --test_size 0.0 --val_size 0.05
```

Train the projection head with proxy anchor loss:
```bash
python src/train.py
```

The default config (`configs/train.yaml`) uses proxy anchor loss with the parameters reported in the paper (margin=0.2, α=32, batch size=1024, AdamW lr=0.001, cosine schedule with 20-epoch warmup, early stopping patience=15). Data paths point to the HPC filesystem and must be updated for local runs.

To override parameters:
```bash
python src/train.py trainer.max_epochs=500 learning.lr=0.0005
```

### Step 5 — Extract projected embeddings and evaluate

```bash
# Extract 128-d projected embeddings from trained checkpoint
python scripts/extract_embeddings.py \
    --checkpoint outputs/.../checkpoints/best.ckpt \
    --lmdb_path embeddings.lmdb \
    --fasta s90_reps.fasta \
    --output_dir output/contrastive/

# Then run hdbscan_benchmark.py on the contrastive .pt file
python scripts/hdbscan_benchmark.py \
    --pt_file output/contrastive/E_starting_contrastive.pt \
    --cluster_tsv s90_cluster.tsv \
    --ec_file ec_annotations.csv \
    --output_dir output/hdbscan_contrastive/
```

---

## Correspondence to report sections

| Report section | Relevant scripts / configs |
|---|---|
| Methods: Embedding generation | `scripts/embed_prostt5.py`, `scripts/embed-esm2.py`, `.qsub` files |
| Methods: HDBSCAN clustering | `scripts/hdbscan_benchmark.py` |
| Methods: Contrastive learning framework | `src/model.py`, `src/losses.py`, `configs/train.yaml`, `configs/experiment/proxy_anchor.yaml` |
| Methods: Training procedure | `src/train.py`, `configs/train.yaml` |
| Methods: Evaluation metrics | `scripts/calculate_ec_purity.py`, `scripts/calculate_ec_purity_and_split_proportion.py` |
| Results: Figures 3–4 (EC purity) | `scripts/hdbscan_benchmark.py` → `scripts/calculate_ec_purity.py` |
| Results: Figure 5 (FunFam sizes) | `scripts/count_funfam_sizes.py` |
| Results: Figure 6 (EC term fragmentation) | `scripts/calculate_ec_purity_and_split_proportion.py` |
| Results: Figure 7, Table 2 (contrastive learning) | Steps 4–5 above |

---

## Model architecture

The projection network is a two-layer MLP applied to frozen pretrained embeddings:

- Input: 1024-dim (ProstT5) or 1280-dim (ESM2-650M) mean-pooled embeddings
- Hidden layer: same dimensionality as input
- Output: 128-dim L2-normalised embeddings
- Dropout: 0.2

Training uses proxy anchor loss (margin=0.2, α=32) with AdamW (lr=0.001, weight decay=1e-4) and a cosine learning rate schedule with 20-epoch linear warmup. Early stopping is applied on validation loss (patience=15, min_delta=1e-4).

---

## HPC job scripts

The `scripts/*.qsub` files are SGE job submission scripts for the UCL HPC cluster (A40/A10 GPUs). They document the exact commands and resource requests used for all embedding and training jobs. They require adaptation of file paths to run outside the UCL environment.
