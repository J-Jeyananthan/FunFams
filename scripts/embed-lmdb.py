#!/usr/bin/env python3
"""
Generate ProstT5 embeddings for protein sequences in FASTA format and store in LMDB,
sharded across multiple GPUs on the same node via hash-by-ID.

Usage example (4 GPUs on one node):
  for r in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$r \
    python embed_prostt5_lmdb_sharded.py \
      -i sequences.fasta \
      -o /path/to/out/embeddings \
      --world_size 4 --rank $r \
      --per_protein 1 --half 1 --storage_dtype float16 \
      --map_size_gb 64 --commit_every 5000 \
      --subdir 1 &
  done
  wait

This will create:
  /path/to/out/embeddings.rank0.lmdb/
  /path/to/out/embeddings.rank1.lmdb/
  /path/to/out/embeddings.rank2.lmdb/
  /path/to/out/embeddings.rank3.lmdb/
"""

import argparse
import hashlib
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Generator, Tuple, Optional

import lmdb
import numpy as np
import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("prostt5_lmdb_sharded")


# ----------------------------
# FASTA streaming utilities
# ----------------------------
def iter_fasta_records(
    fasta_path: Path,
    split_char: str,
    id_field: int,
    is_3Di: bool,
) -> Generator[Tuple[str, str], None, None]:
    """
    Stream FASTA records as (protein_id, sequence).
    - Does NOT load the full FASTA into memory.
    - Applies the same ID cleaning as your original script.
    - Joins multiline sequences.
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
                    seq = "".join(cur_seq_parts)
                    yield cur_id, seq

                header = line[1:].strip()
                parts = header.split(split_char)
                if id_field >= len(parts):
                    # Fallback: whole header if split doesn't have that many fields
                    raw_id = header
                else:
                    raw_id = parts[id_field]

                # Replace tokens that are mis-interpreted when loading h5 (kept from original)
                cur_id = raw_id.replace("/", "_").replace(".", "_")
                cur_seq_parts = []
            else:
                s = "".join(line.split()).replace("-", "")
                if is_3Di:
                    s = s.lower()
                cur_seq_parts.append(s)

    # Emit final record
    if cur_id is not None:
        seq = "".join(cur_seq_parts)
        yield cur_id, seq


# ----------------------------
# Sharding (hash by ID)
# ----------------------------
def shard_for_id(protein_id: str, world_size: int) -> int:
    """
    Stable sharding: md5(id) -> integer -> mod world_size.
    Do NOT use Python's built-in hash() here (it changes per process/run).
    """
    h = hashlib.md5(protein_id.encode("utf-8")).digest()
    # Use first 8 bytes as an integer for speed (still stable)
    n = int.from_bytes(h[:8], byteorder="big", signed=False)
    return n % world_size


# ----------------------------
# Device + model
# ----------------------------
def get_device(rank: int) -> torch.device:
    """
    Choose device, binding to GPU 0 within the process's visible devices.
    Recommended usage: set CUDA_VISIBLE_DEVICES per process.
    """
    if torch.cuda.is_available():
        # If user sets CUDA_VISIBLE_DEVICES to a single GPU, cuda:0 is correct.
        # If they don't, you can still map rank -> cuda:rank.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible:
            # Under CUDA_VISIBLE_DEVICES, rank->cuda:0 typically.
            return torch.device("cuda:0")
        return torch.device(f"cuda:{rank}")
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


def open_lmdb_env(
    path: Path,
    map_size_gb: int,
    subdir: bool,
) -> lmdb.Environment:
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
        readahead=False,   # often better on network FS for writing
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
# Embedding + sharded writing
# ----------------------------
def embed_fasta_to_lmdb_sharded(
    fasta_path: Path,
    out_base: Path,
    model_dir: str,
    split_char: str,
    id_field: int,
    per_protein: bool,
    half_precision: bool,
    is_3Di: bool,
    storage_dtype: str,
    map_size_gb: int,
    commit_every: int,
    max_residues: int,
    max_seq_len: int,
    max_batch: int,
    world_size: int,
    rank: int,
    subdir: bool,
    overwrite: bool,
    write_keys_txt: bool,
):
    assert world_size >= 1
    assert 0 <= rank < world_size

    device = get_device(rank)
    logger.info(f"[rank {rank}/{world_size}] Using device: {device}")
    if device.type == "cuda":
        # Ensure correct device is selected when not using CUDA_VISIBLE_DEVICES trick
        torch.cuda.set_device(device)

    prefix = "<fold2AA>" if is_3Di else "<AA2fold>"

    model, vocab = get_T5_model(model_dir, device)
    if half_precision:
        model = model.half()
        logger.info(f"[rank {rank}] Using model in half-precision")

    if storage_dtype not in ("float16", "float32"):
        raise ValueError("--storage_dtype must be float16 or float32")

    # Output path per shard
    shard_path = out_base.with_name(out_base.name + f".rank{rank}.lmdb")
    logger.info(f"[rank {rank}] Writing LMDB shard to: {shard_path}")

    env = open_lmdb_env(shard_path, map_size_gb=map_size_gb, subdir=subdir)
    db = env.open_db(b"embeddings")
    meta_db = env.open_db(b"meta")

    # Write metadata (includes sharding info)
    meta = {
        "model": model_dir,
        "per_protein": bool(per_protein),
        "half_precision_model": bool(half_precision),
        "is_3Di": bool(is_3Di),
        "storage_dtype": storage_dtype,
        "created_unix": int(time.time()),
        "world_size": world_size,
        "rank": rank,
        "prefix": prefix,
    }
    with env.begin(write=True, db=meta_db) as txn:
        txn.put(b"__meta__", json.dumps(meta).encode("utf-8"))

    keys_fh = None
    if write_keys_txt:
        keys_path = shard_path.with_suffix(shard_path.suffix + ".keys.txt")  # e.g. .lmdb.keys.txt
        keys_fh = open(keys_path, "w")
        logger.info(f"[rank {rank}] Writing keys index to: {keys_path}")

    start = time.time()
    wrote = 0
    skipped_existing = 0
    failed = 0

    # We keep a local batch list
    batch = []

    # LMDB transaction
    txn = env.begin(write=True, db=db)

    def commit_txn():
        nonlocal txn
        txn.commit()
        txn = env.begin(write=True, db=db)

    try:
        # NOTE: we can't know total items for this rank without a first pass
        # so tqdm is "unknown total"; still useful.
        for pid, seq_raw in tqdm(iter_fasta_records(fasta_path, split_char, id_field, is_3Di),
                                 desc=f"Embedding rank {rank}", unit="seq"):
            # Shard selection
            if shard_for_id(pid, world_size) != rank:
                continue

            # Replace non-standard amino acids (from your original script)
            seq = seq_raw.replace("U", "X").replace("Z", "X").replace("O", "X")
            s_len = len(seq)

            # Optional: if you want to skip ultra-long sequences rather than force a 1-seq batch:
            # (kept similar to original behavior; original may still attempt it, then catch OOM)
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
                    # If OOM, we log and skip this whole batch; you could implement
                    # a fallback that retries smaller batch sizes, but keeping it simple.
                    logger.warning(f"[rank {rank}] RuntimeError during embedding batch (last id {pdb_ids[-1]}): {e}")
                    failed += len(pdb_ids)
                    # Try to clear cache on CUDA
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue

                for i, identifier in enumerate(pdb_ids):
                    L = seq_lens[i]
                    # account for prefix token (same slicing approach as your script)
                    emb = out.last_hidden_state[i, 1 : L + 1]

                    if per_protein:
                        emb = emb.mean(dim=0)  # (d,)

                    arr = emb.detach().cpu().numpy().squeeze()

                    if storage_dtype == "float16":
                        arr = arr.astype(np.float16, copy=False)
                    else:
                        arr = arr.astype(np.float32, copy=False)

                    key = identifier.encode("utf-8")

                    if not overwrite:
                        if txn.get(key) is not None:
                            skipped_existing += 1
                            continue

                    val = npy_dumps(arr)

                    # Put with MapFullError handling.
                    while True:
                        try:
                            txn.put(key, val, overwrite=overwrite)
                            break
                        except lmdb.MapFullError:
                            # Must abort this txn to change mapsize
                            txn.abort()
                            grow_map(env)
                            txn = env.begin(write=True, db=db)

                    wrote += 1
                    if keys_fh is not None:
                        keys_fh.write(identifier + "\n")

                    if commit_every > 0 and wrote % commit_every == 0:
                        commit_txn()

        # flush any remaining batch
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
                logger.warning(f"[rank {rank}] RuntimeError during final batch: {e}")
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
                            txn = env.begin(write=True, db=db)

                    wrote += 1
                    if keys_fh is not None:
                        keys_fh.write(identifier + "\n")

        # final commit
        txn.commit()

    except Exception:
        # ensure no txn left open
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
    logger.info(f"[rank {rank}] Wrote embeddings: {wrote}")
    if skipped_existing:
        logger.info(f"[rank {rank}] Skipped existing duplicates: {skipped_existing}")
    if failed:
        logger.info(f"[rank {rank}] Failed (runtime errors): {failed}")
    if wrote:
        logger.info(f"[rank {rank}] Time: {dt:.2f}s | {dt/wrote:.5f} s/protein")


# ----------------------------
# CLI
# ----------------------------
def create_arg_parser():
    p = argparse.ArgumentParser(
        description="Generate ProstT5 embeddings and write to LMDB shards using hash-by-ID across GPUs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("-i", "--input", required=True, type=str, help="Input FASTA file")
    p.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Output base path (shards will be output.rank{r}.lmdb)",
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

    p.add_argument("--max_residues", type=int, default=4000, help="Max residues per batch")
    p.add_argument("--max_seq_len", type=int, default=1000, help="Max sequence length (for stats / batching behavior)")
    p.add_argument("--max_batch", type=int, default=100, help="Max sequences per batch")

    p.add_argument("--world_size", type=int, default=4, help="Number of shards / GPU workers")
    p.add_argument("--rank", type=int, default=0, help="This process rank in [0, world_size-1]")

    p.add_argument("--subdir", type=int, default=1, choices=[0, 1],
                   help="If 1, LMDB is a directory env; if 0, single-file env")
    p.add_argument("--overwrite", type=int, default=0, choices=[0, 1],
                   help="If 1, overwrite keys if they exist; if 0, skip duplicates")
    p.add_argument("--write_keys_txt", type=int, default=1, choices=[0, 1],
                   help="Write a .keys.txt file alongside each shard for fast Dataset indexing")

    return p


def main():
    args = create_arg_parser().parse_args()

    fasta_path = Path(args.input)
    out_base = Path(args.output)

    if not fasta_path.exists():
        raise FileNotFoundError(f"Input FASTA not found: {fasta_path}")

    embed_fasta_to_lmdb_sharded(
        fasta_path=fasta_path,
        out_base=out_base,
        model_dir=args.model,
        split_char=args.split_char,
        id_field=args.id,
        per_protein=bool(args.per_protein),
        half_precision=bool(args.half),
        is_3Di=bool(args.is_3Di),
        storage_dtype=args.storage_dtype,
        map_size_gb=args.map_size_gb,
        commit_every=args.commit_every,
        max_residues=args.max_residues,
        max_seq_len=args.max_seq_len,
        max_batch=args.max_batch,
        world_size=args.world_size,
        rank=args.rank,
        subdir=bool(args.subdir),
        overwrite=bool(args.overwrite),
        write_keys_txt=bool(args.write_keys_txt),
    )


if __name__ == "__main__":
    main()
