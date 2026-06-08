#!/usr/bin/env python3
"""Plot UMAP coords saved by umap_hups_compute.py.

Run this locally after copying the .npz from HPC. Re-run with different
--color_level values to switch EC colouring without re-computing UMAP.

Usage:
    python scripts/umap_hups_plot.py \
        --npz         umap_coords.npz \
        --output      umap_hups_ec1.png \
        [--color_level 1] \
        [--point_size 8] \
        [--dpi 200]
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


EC_CLASS_NAMES = {
    "1": "Oxidoreductases",
    "2": "Transferases",
    "3": "Hydrolases",
    "4": "Lyases",
    "5": "Isomerases",
    "6": "Ligases",
    "7": "Translocases",
}

EC_CLASS_COLOURS = {
    "1": "#e41a1c",
    "2": "#377eb8",
    "3": "#4daf4a",
    "4": "#ff7f00",
    "5": "#984ea3",
    "6": "#a65628",
    "7": "#f781bf",
}

UNKNOWN_COLOUR = "#cccccc"


def ec_colour_label(ec_term_str: str, level: int) -> str:
    """Return the unambiguous EC prefix at `level` digits, or 'unknown'."""
    if not ec_term_str:
        return "unknown"

    ec_set = set(ec_term_str.split("|"))
    prefixes = set()
    for ec in ec_set:
        parts = ec.split(".")
        if len(parts) >= level and all(p != "-" for p in parts[:level]):
            prefixes.add(".".join(parts[:level]))

    if len(prefixes) == 1:
        return prefixes.pop()
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Plot UMAP coords from umap_hups_compute.py.")
    parser.add_argument("--npz",         required=True, help="Path to .npz from umap_hups_compute.py")
    parser.add_argument("--output",      default="umap_hups_ec.png", help="Output image path")
    parser.add_argument("--color_level", type=int, default=1, choices=[1, 2, 3, 4],
                        help="EC level to colour by (default: 1)")
    parser.add_argument("--point_size",  type=float, default=8)
    parser.add_argument("--dpi",         type=int, default=200)
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    coords = data["coords"]
    ec_terms = data["ec_terms"].tolist()

    labels = [ec_colour_label(t, args.color_level) for t in ec_terms]
    n_coloured = sum(1 for l in labels if l != "unknown")
    print(f"{n_coloured:,} / {len(labels):,} reps have unambiguous EC at level {args.color_level}")

    unknown_mask = np.array([l == "unknown" for l in labels])
    present_labels = sorted(set(labels) - {"unknown"})

    # Level 1: fixed palette. Higher levels: shades of the parent EC class colour.
    if args.color_level == 1:
        colour_map = EC_CLASS_COLOURS
    else:
        # Group sub-labels by their EC class 1 parent
        from matplotlib.colors import to_rgb
        import colorsys

        by_class: dict[str, list[str]] = defaultdict(list)
        for label in present_labels:
            by_class[label.split(".")[0]].append(label)

        colour_map = {}
        for ec_class, sublabels in by_class.items():
            base = EC_CLASS_COLOURS.get(ec_class, "#888888")
            r, g, b = to_rgb(base)
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            n = len(sublabels)
            for i, label in enumerate(sorted(sublabels)):
                # Vary saturation only (pastel → vivid), keeping hue and value
                # fixed so adjacent EC classes don't bleed into each other
                shade_s = s * (0.25 + 0.75 * (i / max(n - 1, 1)))
                colour_map[label] = colorsys.hsv_to_rgb(h, shade_s, v)

    fig, ax = plt.subplots(figsize=(10, 8))

    if unknown_mask.any():
        ax.scatter(
            coords[unknown_mask, 0], coords[unknown_mask, 1],
            c=UNKNOWN_COLOUR, s=args.point_size, alpha=0.4, linewidths=0,
            rasterized=True,
        )

    for label in present_labels:
        mask = np.array([l == label for l in labels])
        colour = colour_map.get(label, UNKNOWN_COLOUR)
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[colour], s=args.point_size, alpha=0.8, linewidths=0,
            rasterized=True,
        )

    handles = [mpatches.Patch(color=UNKNOWN_COLOUR, label="unknown / ambiguous")]
    if args.color_level == 1:
        for label in present_labels:
            colour = colour_map.get(label, UNKNOWN_COLOUR)
            name = f"EC {label} — {EC_CLASS_NAMES[label]}"
            handles.append(mpatches.Patch(color=colour, label=name))
    else:
        # Show one swatch per EC class 1 group rather than every sub-label
        for ec_class in sorted(by_class):
            base = EC_CLASS_COLOURS.get(ec_class, "#888888")
            name = f"EC {ec_class} — {EC_CLASS_NAMES.get(ec_class, ec_class)} ({len(by_class[ec_class])} subclasses)"
            handles.append(mpatches.Patch(color=base, label=name))

    ax.legend(handles=handles, loc="best", markerscale=2, fontsize=9,
               framealpha=0.8, title=f"EC level {args.color_level}")

    ax.set_title(
        f"UMAP — HUPs S90 cluster reps (n={len(labels):,}, {n_coloured:,} with EC)\n"
        f"ProstT5, L2-normalised, coloured by EC level {args.color_level}",
        fontsize=12,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
