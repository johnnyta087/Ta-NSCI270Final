#!/usr/bin/env python3
"""Export each results graph as its own PNG, without titles.

Reuses the data loading and label settings from build_results_viz.py, so the numbers are the same
as in results_visualization.html.
"""
import csv
import os
import sys

sys.path.insert(0, "/hickory/users/j/o/johnnyta/nnunet/tools")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import build_results_viz as B

OUTD = os.path.join(B.OUT, "png_exports")
TEST = "Held-out test set (51 cases, scored once)"
DEV = "Out-of-fold development set (204 cases)"


def save(fig, name):
    fig.savefig(os.path.join(OUTD, name), dpi=200, bbox_inches="tight")
    plt.close(fig)


def region_values(cases_csv):
    vals = {r: {"dice": [], "hd95": []} for r in ["WT", "TC", "ET"]}
    for row in csv.DictReader(open(cases_csv)):
        if row["status"] != "both_empty":
            vals[row["region"]]["dice"].append(float(row["dice"]))
        if row["hd95_mm"] not in ("", "nan"):
            vals[row["region"]]["hd95"].append(float(row["hd95_mm"]))
    return vals


def region_plot(metric, ylabel, fname):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    data, labels = [], []
    for setname, short in [(TEST, "Test"), (DEV, "Dev")]:
        vals = region_values(B.SETS[setname]["cases"])
        for r in ["WT", "TC", "ET"]:
            data.append(vals[r][metric])
            labels.append(f"{r}\n{short}")
    ax.boxplot(data, widths=0.6)
    ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=9)
    ax.set_ylabel(ylabel)
    if metric == "dice":
        ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", alpha=0.3)
    save(fig, fname)


def label_plot(setname, fname):
    vals = B.label_dice(B.SETS[setname]["summary"])
    keys = [l for l in B.LABEL_NAMES if vals[l]]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.boxplot([vals[l] for l in keys], widths=0.6)
    for i, l in enumerate(keys, start=1):
        x = np.random.default_rng(l).normal(i, 0.05, size=len(vals[l]))
        ax.scatter(x, vals[l], s=9, alpha=0.5, color=B.LABEL_COLOURS[l])
    ax.set_xticks(range(1, len(keys) + 1), [B.LABEL_NAMES[l] for l in keys], fontsize=8)
    ax.set_ylabel("Dice (cases where label is present)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", alpha=0.3)
    save(fig, fname)


def overlay_plot(tag, cid, seq):
    gt = B.load(f"{B.NN}/holdout_test/labels/{cid}.nii.gz").astype(int)
    pr = B.load(f"{B.NN}/scoring/holdout_502_pred/{cid}.nii.gz").astype(int)
    img = B.load(f"{B.PKG}/{cid}/{cid}-{seq}.nii.gz").astype(float)
    z = int(np.argmax((gt > 0).sum(axis=(0, 1))))
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    sl = np.rot90(img[:, :, z])
    lo, hi = np.percentile(img[img > 0], [1, 99.5]) if (img > 0).any() else (0, 1)
    ax.imshow(sl, cmap="gray", vmin=lo, vmax=hi)
    ax.axis("off")
    for l, col in B.LABEL_COLOURS.items():
        g = np.rot90((gt[:, :, z] == l).astype(float))
        p = np.rot90((pr[:, :, z] == l).astype(float))
        if g.any():
            ax.contour(g, levels=[0.5], colors=[col], linewidths=1.5, linestyles="solid")
        if p.any():
            ax.contour(p, levels=[0.5], colors=[col], linewidths=1.2, linestyles="dashed")
    handles = [Line2D([0], [0], color=c, lw=2, label=B.LABEL_NAMES[l]) for l, c in B.LABEL_COLOURS.items()]
    handles += [Line2D([0], [0], color="k", lw=1.5, ls="solid", label="expert"),
                Line2D([0], [0], color="k", lw=1.5, ls="dashed", label="model")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    save(fig, f"analysis3_{tag}_{cid}_{seq}.png")


def main():
    os.makedirs(OUTD, exist_ok=True)
    region_plot("dice", "Dice", "analysis1_region_dice.png")
    region_plot("hd95", "HD95 (mm)", "analysis1_region_hd95.png")
    label_plot(TEST, "analysis2_labels_test.png")
    label_plot(DEV, "analysis2_labels_development.png")
    for tag, cid, dice in B.pick_cases(B.SETS[TEST]["cases"]):
        overlay_plot(tag, cid, "t2f")
        overlay_plot(tag, cid, "t1c")
    print("exported to", OUTD)
    for f in sorted(os.listdir(OUTD)):
        print(f)


if __name__ == "__main__":
    main()
