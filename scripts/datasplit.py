import argparse
from pathlib import Path
from Bio import SeqIO
from sklearn.model_selection import train_test_split



def label_loader(records, mapping_file: Path) -> list[str]:
    '''Produces a list of labels corresponding to the sequences in the input fasta. 
       Assumes that the mapping file is in the format 'domain_id<tab>funfam_label' per line, e.g 'Q8EJT8/3-88	1.10.10.10-FF-000001', 
       and that fasta headers are domain IDs only, e.g 'Q8EJT8/3-88'.
       '''

    label_map = {}
    with open(mapping_file) as f:
        for line in f:
            cath_id, label = line.rstrip("\n").split("\t")
            label_map[cath_id] = label

    print(f"Number of labels in label_map: {len(label_map)}")

    labels = []

    for record in records:
        if record.id not in label_map:
            raise ValueError(f"Missing label for {record.id}")
        
        labels.append(label_map[record.id])

    print(f"Number of records with labels: {len(labels)}, example label : {labels[0]}")
    return labels

def write_labels(records, labels, out_path: Path):
    with open(out_path, "w") as f:
        for record, label in zip(records, labels):
            f.write(f"{record.id}\t{label}\n")

def dataset_splitter(fasta_file: Path, mapping_file: Path, output_dir: Path, test_size: float, val_size: float, random_seed: int = 42):
    '''Performs a stratified split of the dataset into train, validation and test sets based on the provided sizes. 
       Assumes that each label occurs at least 3 times in the dataset to allow for stratified splitting.
       Saves the splits as separate fasta files in the output directory.
       '''
    if not (0 < test_size < 1) or not (0 < val_size < 1) or (test_size + val_size >= 1):
        raise ValueError("Require 0<test_size<1, 0<val_size<1, and test_size+val_size<1")
    
    records = list(SeqIO.parse(fasta_file, "fasta"))
    labels = label_loader(records, mapping_file)

    # First split off the test set
    train_val, test, train_val_labels, test_labels = train_test_split(
        records, labels, test_size=test_size, stratify=labels, random_state=random_seed
    )

    # Then split the remaining data into training and validation sets
    val_relative_size = val_size / (1 - test_size)
    train, val, train_labels, val_labels = train_test_split(
        train_val, train_val_labels, test_size=val_relative_size, stratify=train_val_labels, random_state=random_seed
    )

    # Save the splits
    output_dir.mkdir(parents=True, exist_ok=True)
    SeqIO.write(train, output_dir / "train.fasta", "fasta")
    SeqIO.write(val, output_dir / "val.fasta", "fasta")
    SeqIO.write(test, output_dir / "test.fasta", "fasta")

    write_labels(train, train_labels, output_dir / "train_labels.txt")
    write_labels(val, val_labels, output_dir / "val_labels.txt")
    write_labels(test, test_labels, output_dir / "test_labels.txt")

    print(f"Saved {len(train)} training sequences to {output_dir / 'train.fasta'}")
    print(f"Saved {len(val)} validation sequences to {output_dir / 'val.fasta'}")
    print(f"Saved {len(test)} test sequences to {output_dir / 'test.fasta'}")
    print(f"Saved corresponding label files to {output_dir}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train, validation, and test sets.")
    parser.add_argument("--fasta_file", type=Path, required=True, help="Path to the input fasta file.")
    parser.add_argument("--mapping_file", type=Path, required=True, help="Path to the mapping file (domain_id to funfam_label).")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save the output splits.")
    parser.add_argument("--test_size", type=float, required=True, help="Proportion of the dataset to include in the test split.")
    parser.add_argument("--val_size", type=float, required=True, help="Proportion of the dataset to include in the validation split.")
    

    args = parser.parse_args()

    dataset_splitter(
        fasta_file=args.fasta_file,
        mapping_file=args.mapping_file,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
    )


