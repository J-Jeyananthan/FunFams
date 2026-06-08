#!/usr/bin/env python3
"""
Generate ProstT5 embeddings for protein sequences in FASTA format and store in a single LMDB.

- Single GPU (or CPU if no CUDA)
- Single LMDB environment (directory by default)
- Streams FASTA (does NOT load all sequences into memory)
- Stores each embedding as a NumPy .npy blob (dtype+shape preserved)
- Periodic commits
- Auto-grows LMDB map size on MapFullError

Example:
  python embed_lmdb_single.py \
    -i sequences.fasta \
    -o embeddings.lmdb \
    --per_protein 1 --half 1 --storage_dtype float16 \
    --map_size_gb 64 --commit_every 5000
"""

import argparse
import io
import json
import logging
import time
from pathlib import Path
from typing import Generator, Optional, Tuple

import lmdb
import numpy as np
import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("prostt5_lmdb_single")


# ----------------------------
# FASTA streaming
# ----------------------------
def iter_fasta_records(
    fasta_path: Path,
    split_char: str,
    id_field: int,
    is_3Di: bool,
) -> Generator[Tuple[str, str], None, None]:
    """
    Stream FASTA records as (protein_id, sequence).

    - Extracts ID from header using split_char and id_field.
    - Applies same ID cleaning as your original script: '/' and '.' -> '_'.
    - Joins multiline sequences, removes '-' and whitespace.
    - Lowercases if is_3Di.
    """
    cur_id: Optional[str] = None
    cur_seq_parts = []

    with open(fasta_path, "r") as f:
        for line in f:
            if not line:
                continue

            if line.startswith(">"):
                # Emit previous record
                if cur_id is not None:
                    yield cur_id, "".join(cur_seq_parts)

                header = line[1:].strip()
                parts = header.split(split_char)
                raw_id = parts[id_field] if id_field < len(parts) else header

                cur_id = raw_id.replace("/", "_").replace(".", "_")
                cur_seq_parts = []
            else:
                s = "".join(line.split()).replace("-", "")
                if is_3Di:
                    s = s.lower()
                cur_seq_parts.append(s)

    # Emit final record
    if cur_id is not None:
        yield cur_id, "".join(cur_seq_parts)


# ----------------------------
# Model / device
# ----------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_T5_model(model_dir: str, device: torch.device):
    logger.info(f"Loading T5 from: {model_dir}")
    model = T5EncoderModel.from_pretrained(model_dir).to(device).eval()
    vocab = T5Tokenizer.from_pretrained(model_dir, do_lower_case=False)
    return model, vocab


# ----------------------------
# LMDB helpers
# ----------------------------
def npy_dumps(array: np.ndarray) -> bytes:
    """Serialize numpy array into .npy bytes (dtype+shape included, portable)."""
    bio = io.BytesIO()
    np.save(bio, array, allow_pickle=False)
    return bio.getvalue()


def open_lmdb_env(path: Path, map_size_gb: int, subdir: bool) -> lmdb.Environment:
    if subdir:
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(
        str(path),
        map_size=int(map_size_gb) * (1024**3),
        subdir=subdir,
        readonly=False,
        lock=True,
        readahead=False,   # typically good for writers on network FS
        meminit=False,
        max_dbs=2,
    )
    return env


def grow_map(env: lmdb.Environment, min_grow_bytes: int = 1 * (1024**3)) -> None:
    info = env.info()
    cur = int(info.get("map_size", 0))
    new = max(cur * 2, cur + min_grow_bytes)
    logger.warning(f"LMDB MapFullError: growing map from {cur/1024**3:.1f} GB -> {new/1024**3:.1f} GB")
    env.set_mapsize(new)


# ----------------------------
# Main embedding loop
# ----------------------------
def embed_fasta_to_lmdb(
    fasta_path: Path,
    out_lmdb: Path,
    model_dir: str,
    split_char: str,
    id_field: int,
    per_protein: bool,
    half_precision: bool,
    is_3Di: bool,
    storage_dtype: str,
    map_size_gb: int,
    commit_every: int,
    write_keys_txt: bool,
    subdir: bool,
    overwrite: bool,
    max_residues: int,
    max_seq_len: int,
    max_batch: int,
) -> None:
    device = get_device()
    logger.info(f"Using device: {device}")

    prefix = "<fold2AA>" if is_3Di else "<AA2fold>"

    model, vocab = get_T5_model(model_dir, device)
    if half_precision:
        model = model.half()
        logger.info("Using model in half-precision")

    if storage_dtype not in ("float16", "float32"):
        raise ValueError("--storage_dtype must be float16 or float32")

    logger.info(f"Writing LMDB to: {out_lmdb}")
    env = open_lmdb_env(out_lmdb, map_size_gb=map_size_gb, subdir=subdir)
    emb_db = env.open_db(b"embeddings")
    meta_db = env.open_db(b"meta")

    meta = {
        "model": model_dir,
        "per_protein": bool(per_protein),
        "half_precision_model": bool(half_precision),
        "is_3Di": bool(is_3Di),
        "storage_dtype": storage_dtype,
        "created_unix": int(time.time()),
        "prefix": prefix,
    }
    with env.begin(write=True, db=meta_db) as txn:
        txn.put(b"__meta__", json.dumps(meta).encode("utf-8"))

    keys_fh = None
    if write_keys_txt:
        keys_path = out_lmdb.with_suffix(out_lmdb.suffix + ".keys.txt")  # e.g. embeddings.lmdb.keys.txt
        keys_fh = open(keys_path, "w")
        logger.info(f"Writing keys index to: {keys_path}")

    start = time.time()
    wrote = 0
    skipped_existing = 0
    failed = 0

    batch = []
    txn = env.begin(write=True, db=emb_db)

    def commit_txn():
        nonlocal txn
        txn.commit()
        txn = env.begin(write=True, db=emb_db)

    try:
        for pid, seq_raw in tqdm(iter_fasta_records(fasta_path, split_char, id_field, is_3Di),
                                 desc="Embedding", unit="seq"):
            # Replace non-standard amino acids
            seq = seq_raw.replace("U", "X").replace("Z", "X").replace("O", "X")
            s_len = len(seq)

            # Build token string: prefix + spaced characters
            seq_tok = prefix + " " + " ".join(list(seq))
            batch.append((pid, seq_tok, s_len))

            n_res_batch = sum(x[2] for x in batch)
            if len(batch) >= max_batch or n_res_batch >= max_residues or s_len > max_seq_len:
                pdb_ids, seqs, seq_lens = zip(*batch)
                batch = []

                token_encoding = vocab.batch_encode_plus(
                    seqs,
                    add_special_tokens=True,
                    padding="longest",
                    return_tensors="pt",
                ).to(device)

                try:
                    with torch.no_grad():
                        out = model(
                            token_encoding.input_ids,
                            attention_mask=token_encoding.attention_mask,
                        )
                except RuntimeError as e:
                    logger.warning(f"RuntimeError during embedding batch (last id {pdb_ids[-1]}): {e}")
                    failed += len(pdb_ids)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue

                for i, identifier in enumerate(pdb_ids):
                    L = seq_lens[i]
                    emb = out.last_hidden_state[i, 1 : L + 1]  # account for prefix token

                    if per_protein:
                        emb = emb.mean(dim=0)  # (d,)

                    arr = emb.detach().cpu().numpy().squeeze()
                    arr = arr.astype(np.float16 if storage_dtype == "float16" else np.float32, copy=False)

                    key = identifier.encode("utf-8")

                    if not overwrite and txn.get(key) is not None:
                        skipped_existing += 1
                        continue

                    val = npy_dumps(arr)

                    while True:
                        try:
                            txn.put(key, val, overwrite=overwrite)
                            break
                        except lmdb.MapFullError:
                            txn.abort()
                            grow_map(env)
                            txn = env.begin(write=True, db=emb_db)

                    wrote += 1
                    if keys_fh is not None:
                        keys_fh.write(identifier + "\n")

                    if wrote == 1:
                        logger.info(f"Example embedded {identifier} -> dtype {arr.dtype}, shape {arr.shape}")

                    if commit_every > 0 and wrote % commit_every == 0:
                        commit_txn()

        # Flush remaining batch
        if batch:
            pdb_ids, seqs, seq_lens = zip(*batch)

            token_encoding = vocab.batch_encode_plus(
                seqs,
                add_special_tokens=True,
                padding="longest",
                return_tensors="pt",
            ).to(device)

            try:
                with torch.no_grad():
                    out = model(
                        token_encoding.input_ids,
                        attention_mask=token_encoding.attention_mask,
                    )
            except RuntimeError as e:
                logger.warning(f"RuntimeError during final batch: {e}")
                failed += len(pdb_ids)
                out = None

            if out is not None:
                for i, identifier in enumerate(pdb_ids):
                    L = seq_lens[i]
                    emb = out.last_hidden_state[i, 1 : L + 1]
                    if per_protein:
                        emb = emb.mean(dim=0)

                    arr = emb.detach().cpu().numpy().squeeze()
                    arr = arr.astype(np.float16 if storage_dtype == "float16" else np.float32, copy=False)

                    key = identifier.encode("utf-8")
                    if not overwrite and txn.get(key) is not None:
                        skipped_existing += 1
                        continue

                    val = npy_dumps(arr)
                    while True:
                        try:
                            txn.put(key, val, overwrite=overwrite)
                            break
                        except lmdb.MapFullError:
                            txn.abort()
                            grow_map(env)
                            txn = env.begin(write=True, db=emb_db)

                    wrote += 1
                    if keys_fh is not None:
                        keys_fh.write(identifier + "\n")

        txn.commit()

    except Exception:
        try:
            txn.abort()
        except Exception:
            pass
        raise
    finally:
        if keys_fh is not None:
            keys_fh.flush()
            keys_fh.close()
        env.sync()
        env.close()

    dt = time.time() - start
    logger.info(f"Wrote embeddings: {wrote}")
    if skipped_existing:
        logger.info(f"Skipped existing duplicates: {skipped_existing}")
    if failed:
        logger.info(f"Failed (runtime errors): {failed}")
    if wrote:
        logger.info(f"Time: {dt:.2f}s | {dt/wrote:.5f} s/protein")


# ----------------------------
# CLI
# ----------------------------
def create_arg_parser():
    p = argparse.ArgumentParser(
        description="Generate ProstT5 embeddings and write to a single LMDB (no sharding).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("-i", "--input", required=True, type=str, help="Input FASTA file")
    p.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Output LMDB path (directory by default; file if --subdir 0)",
    )

    p.add_argument("--model", type=str, default="Rostlab/ProstT5", help="Model dir or HF id")
    p.add_argument("--split_char", type=str, default="!", help="Split character for FASTA header")
    p.add_argument("--id", type=int, default=0, help="Field index after split for ID extraction")

    p.add_argument("--per_protein", type=int, default=1, choices=[0, 1], help="Mean-pooled per-protein if 1")
    p.add_argument("--half", type=int, default=1, choices=[0, 1], help="Use half-precision model weights if 1")
    p.add_argument("--is_3Di", type=int, default=0, choices=[0, 1], help="Input sequences are 3Di if 1")

    p.add_argument("--storage_dtype", type=str, default="float16", choices=["float16", "float32"],
                   help="Dtype to store embeddings in LMDB")
    p.add_argument("--map_size_gb", type=int, default=64, help="Initial LMDB map size (auto-grows)")
    p.add_argument("--commit_every", type=int, default=5000, help="Commit LMDB every N records")
    p.add_argument("--write_keys_txt", type=int, default=1, choices=[0, 1],
                   help="Write a .keys.txt file alongside the LMDB for fast Dataset indexing")

    p.add_argument("--subdir", type=int, default=1, choices=[0, 1],
                   help="If 1, LMDB is a directory env; if 0, single-file env")
    p.add_argument("--overwrite", type=int, default=0, choices=[0, 1],
                   help="If 1, overwrite keys if they exist; if 0, skip duplicates")

    # batching knobs
    p.add_argument("--max_residues", type=int, default=4000, help="Max residues per batch")
    p.add_argument("--max_seq_len", type=int, default=1000, help="Max sequence length")
    p.add_argument("--max_batch", type=int, default=100, help="Max sequences per batch")

    return p


def main():
    args = create_arg_parser().parse_args()

    fasta_path = Path(args.input)
    out_lmdb = Path(args.output)

    if not fasta_path.exists():
        raise FileNotFoundError(f"Input FASTA not found: {fasta_path}")

    embed_fasta_to_lmdb(
        fasta_path=fasta_path,
        out_lmdb=out_lmdb,
        model_dir=args.model,
        split_char=args.split_char,
        id_field=args.id,
        per_protein=bool(args.per_protein),
        half_precision=bool(args.half),
        is_3Di=bool(args.is_3Di),
        storage_dtype=args.storage_dtype,
        map_size_gb=args.map_size_gb,
        commit_every=args.commit_every,
        write_keys_txt=bool(args.write_keys_txt),
        subdir=bool(args.subdir),
        overwrite=bool(args.overwrite),
        max_residues=args.max_residues,
        max_seq_len=args.max_seq_len,
        max_batch=args.max_batch,
    )


if __name__ == "__main__":
    main()
