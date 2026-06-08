#!/usr/bin/env python3
"""Generate ProstT5 embeddings for protein sequences in FASTA format."""

import argparse
import logging
import time
from pathlib import Path
import torch
import h5py
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_device():
    """Determine available device."""
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_T5_model(model_dir, device):
    """Load T5 encoder model and tokenizer."""
    logger.info(f"Loading T5 from: {model_dir}")
    model = T5EncoderModel.from_pretrained(model_dir).to(device)
    model = model.eval()
    vocab = T5Tokenizer.from_pretrained(model_dir, do_lower_case=False)
    return model, vocab


def read_fasta(fasta_path, split_char, id_field, is_3Di):
    """Read FASTA file and return dictionary of sequences.
    
    Args:
        fasta_path: Path to FASTA file
        split_char: Character to split header for ID extraction
        id_field: Field index after splitting
        is_3Di: Whether sequences are 3Di (structure) or AA (amino acid)
        
    Returns:
        Dictionary mapping sequence IDs to sequences
    """
    sequences = {}
    with open(fasta_path, 'r') as fasta_f:
        for line in fasta_f:
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
                # Replace tokens that are mis-interpreted when loading h5
                uniprot_id = uniprot_id.replace("/", "_").replace(".", "_")
                sequences[uniprot_id] = ''
            else:
                # Replace whitespace and join sequences spanning multiple lines
                if is_3Di:
                    sequences[uniprot_id] += ''.join(line.split()).replace("-", "").lower()
                else:
                    sequences[uniprot_id] += ''.join(line.split()).replace("-", "")
    return sequences


def get_embeddings(seq_path, emb_path, model_dir, split_char, id_field,
                   per_protein, half_precision, is_3Di,
                   max_residues=4000, max_seq_len=1000, max_batch=100):
    """Generate embeddings for sequences in FASTA file.
    
    Args:
        seq_path: Path to input FASTA file
        emb_path: Path to output HDF5 file
        model_dir: Path or HuggingFace model identifier
        split_char: Character to split FASTA headers
        id_field: Field index for sequence ID extraction
        per_protein: If True, return mean-pooled per-protein embeddings
        half_precision: If True, use FP16 precision
        is_3Di: If True, sequences are 3Di structure sequences
        max_residues: Maximum residues per batch
        max_seq_len: Maximum sequence length
        max_batch: Maximum sequences per batch
    """
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Read FASTA file
    seq_dict = read_fasta(seq_path, split_char, id_field, is_3Di)
    prefix = "<fold2AA>" if is_3Di else "<AA2fold>"
    
    model, vocab = get_T5_model(model_dir, device)
    if half_precision:
        model = model.half()
        logger.info("Using model in half-precision")

    logger.info(f"Input is 3Di: {is_3Di}")
    logger.info(f"Total number of sequences: {len(seq_dict)}")

    avg_length = sum(len(seq) for seq in seq_dict.values()) / len(seq_dict)
    n_long = sum(1 for seq in seq_dict.values() if len(seq) > max_seq_len)
    # Sort sequences by length to trigger OOM at the beginning
    seq_dict = sorted(seq_dict.items(), key=lambda kv: len(kv[1]), reverse=True)
    
    logger.info(f"Average sequence length: {avg_length:.1f}")
    logger.info(f"Number of sequences >{max_seq_len}: {n_long}")
    
    start = time.time()
    emb_dict = {}
    batch = []
    
    for seq_idx, (pdb_id, seq) in enumerate(tqdm(seq_dict, total=len(seq_dict), desc="Embedding"), 1):
        # Replace non-standard amino acids
        seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
        seq_len = len(seq)
        seq = prefix + ' ' + ' '.join(list(seq))
        batch.append((pdb_id, seq, seq_len))

        # Count residues in current batch
        n_res_batch = sum(s_len for _, _, s_len in batch)
        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = []

            token_encoding = vocab.batch_encode_plus(
                seqs,
                add_special_tokens=True,
                padding="longest",
                return_tensors='pt'
            ).to(device)
            
            try:
                with torch.no_grad():
                    embedding_repr = model(
                        token_encoding.input_ids,
                        attention_mask=token_encoding.attention_mask
                    )
            except RuntimeError as e:
                logger.warning(f"RuntimeError during embedding for {pdb_id} (L={seq_len}): {e}")
                continue
            
            # Extract embeddings (accounting for prefix token)
            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]
                emb = embedding_repr.last_hidden_state[batch_idx, 1:s_len+1]
                
                if per_protein:
                    emb = emb.mean(dim=0)
                emb_dict[identifier] = emb.detach().cpu().numpy().squeeze()
                
                if len(emb_dict) == 1:
                    logger.info(f"Example: embedded protein {identifier} (L={s_len}) -> shape {emb.shape}")

    end = time.time()
    
    # Save embeddings to HDF5
    with h5py.File(str(emb_path), "w") as hf:
        for sequence_id, embedding in emb_dict.items():
            hf.create_dataset(sequence_id, data=embedding)

    logger.info(f"Total embeddings: {len(emb_dict)}")
    logger.info(f"Total time: {end-start:.2f}s; time/prot: {(end-start)/len(emb_dict):.4f}s; avg len: {avg_length:.2f}")
    return True


def create_arg_parser():
    """Create and return ArgumentParser."""
    parser = argparse.ArgumentParser(
        description='Generate ProstT5-Encoder embeddings for protein sequences in FASTA format.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example: python embed.py -i sequences.fasta -o embeddings.h5 --half 1 --is_3Di 0 --per_protein 1'
    )
    
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='Path to FASTA-formatted file containing protein sequence(s)')
    parser.add_argument('-o', '--output', required=True, type=str,
                        help='Path for saving embeddings as HDF5 file')
    parser.add_argument('--model', type=str, default="Rostlab/ProstT5",
                        help='Path to model directory or HuggingFace model identifier (default: Rostlab/ProstT5)')
    parser.add_argument('--split_char', type=str, default='!',
                        help='Character for splitting FASTA header to extract protein ID (default: !)')
    parser.add_argument('--id', type=int, default=0,
                        help='Field index for sequence ID after splitting header (default: 0)')
    parser.add_argument('--per_protein', type=int, default=0, choices=[0, 1],
                        help='Return per-residue (0) or mean-pooled per-protein (1) embeddings (default: 0)')
    parser.add_argument('--half', type=int, default=0, choices=[0, 1],
                        help='Use half precision (FP16) if 1, full precision if 0 (default: 0)')
    parser.add_argument('--is_3Di', type=int, default=0, choices=[0, 1],
                        help='Input is 3Di structure (1) or amino acid (0) sequences (default: 0)')
    
    return parser


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    
    seq_path = Path(args.input)
    emb_path = Path(args.output)
    
    if not seq_path.exists():
        logger.error(f"Input file not found: {seq_path}")
        return
    
    get_embeddings(
        seq_path,
        emb_path,
        args.model,
        args.split_char,
        args.id,
        per_protein=bool(args.per_protein),
        half_precision=bool(args.half),
        is_3Di=bool(args.is_3Di)
    )


if __name__ == '__main__':
    main()