#!/usr/bin/env python3
"""Combine the three analyses into one PNG each, with labels and legends and no titles.

analysis1_region.png  : Dice and HD95 for WT, TC, ET, held-out test vs development
analysis2_labels.png  : per-label Dice (ET, NET, CC, ED), development and held-out test
analysis3_overlays.png: expert (solid) vs model (dashed) outlines, 3 held-out cases x 2 sequences
"""
import os
import sys

sys.path.insert(0, "/hickory/users/j/o/johnnyta/nnunet/tools")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import build_results_viz as B
import export_pngs as E

OUT = B.OUT
TEST_COL = "#4C72B0"
DEV_COL = "#DD8452"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight")
    plt.close(fig)


def analysis1():
    sets = [(E.TEST, "Held-out test (n = 51)", TEST_COL), (E.DEV, "Development (n = 204)", DEV_COL)]
    vals = {name: E.region_values(B.SETS[name]["cases"]) for name, _, _ in sets}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (metric, ylabel) in zip(axes, [("dice", "Dice"), ("hd95", "HD95 (mm)")]):
        data, pos, cols = [], [], []
        for r_i, region in enumerate(["WT", "TC", "ET"]):
            for g, (name, _, col) in enumerate(sets):
                data.append(vals[name][region][metric])
                pos.append(r_i * 3 + g + 1)
                cols.append(col)
        bp = ax.boxplot(data, positions=pos, widths=0.7, patch_artist=True)
        for patch, col in zip(bp["boxes"], cols):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)
        ax.set_xticks([1.5, 4.5, 7.5], ["WT", "TC", "ET"], fontsize=10)
        ax.set_ylabel(ylabel)
        if metric == "dice":
            ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", alpha=0.3)
    handles = [Patch(facecolor=col, alpha=0.75, label=lab) for _, lab, col in sets]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save(fig, "analysis1_region.png")


def analysis2():
    panels = [(E.DEV, "Development (n = 204)"), (E.TEST, "Held-out test (n = 51)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, (name, lab) in zip(axes, panels):
        vals = B.label_dice(B.SETS[name]["summary"])
        keys = [l for l in B.LABEL_NAMES if vals[l]]
        bp = ax.boxplot([vals[l] for l in keys], widths=0.6, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#e6e6e6")
        for i, l in enumerate(keys, start=1):
            x = np.random.default_rng(l).normal(i, 0.05, size=len(vals[l]))
            ax.scatter(x, vals[l], s=9, alpha=0.6, color=B.LABEL_COLOURS[l])
        ax.set_xticks(range(1, len(keys) + 1),
                      [f"{B.LABEL_NAMES[l]}\nn = {len(vals[l])}" for l in keys], fontsize=8)
        ax.text(0.02, 0.98, lab, transform=ax.transAxes, va="top", fontsize=9, weight="bold")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Dice (cases where label is present)")
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8,
                      label=B.LABEL_NAMES[l]) for l, c in B.LABEL_COLOURS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save(fig, "analysis2_labels.png")


def analysis3():
    picks = B.pick_cases(B.SETS[E.TEST]["cases"])
    fig, axes = plt.subplots(len(picks), 2, figsize=(8.5, 4.4 * len(picks)))
    for r, (tag, cid, dice) in enumerate(picks):
        gt = B.load(f"{B.NN}/holdout_test/labels/{cid}.nii.gz").astype(int)
        pr = B.load(f"{B.NN}/scoring/holdout_502_pred/{cid}.nii.gz").astype(int)
        z = int(np.argmax((gt > 0).sum(axis=(0, 1))))
        for c, (seq, name) in enumerate([("t2f", "T2-FLAIR"), ("t1c", "T1 post-contrast")]):
            ax = axes[r, c]
            img = B.load(f"{B.PKG}/{cid}/{cid}-{seq}.nii.gz").astype(float)
            lo, hi = np.percentile(img[img > 0], [1, 99.5]) if (img > 0).any() else (0, 1)
            ax.imshow(np.rot90(img[:, :, z]), cmap="gray", vmin=lo, vmax=hi)
            for l, col in B.LABEL_COLOURS.items():
                g = np.rot90((gt[:, :, z] == l).astype(float))
                p = np.rot90((pr[:, :, z] == l).astype(float))
                if g.any():
                    ax.contour(g, levels=[0.5], colors=[col], linewidths=1.5, linestyles="solid")
                if p.any():
                    ax.contour(p, levels=[0.5], colors=[col], linewidths=1.2, linestyles="dashed")
            ax.axis("off")
            if r == 0:
                ax.text(0.5, 1.02, name, transform=ax.transAxes, ha="center", va="bottom", fontsize=10)
        axes[r, 0].text(-0.06, 0.5, f"{tag.capitalize()}\nWT Dice {dice:.2f}\n{cid}",
                        transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=8)
    handles = [Line2D([0], [0], color=c, lw=2, label=B.LABEL_NAMES[l]) for l, c in B.LABEL_COLOURS.items()]
    handles += [Line2D([0], [0], color="k", lw=1.5, ls="solid", label="Expert"),
                Line2D([0], [0], color="k", lw=1.5, ls="dashed", label="Model")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.22, wspace=0.05, hspace=0.12, bottom=0.1, top=0.95)
    save(fig, "analysis3_overlays.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    analysis1()
    analysis2()
    analysis3()
    for f in ["analysis1_region.png", "analysis2_labels.png", "analysis3_overlays.png"]:
        print(os.path.join(OUT, f))


if __name__ == "__main__":
    main()
