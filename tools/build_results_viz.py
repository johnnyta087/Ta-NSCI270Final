#!/usr/bin/env python3
"""Build the results visualisation report for the three analyses.

Analysis 1: region accuracy (WT, TC, ET) against expert labels, Dice and HD95.
Analysis 2: per-label Dice (labels 1-4), for the 204 out-of-fold and 51 held-out cases.
Analysis 3: visual comparison of predicted vs expert outlines on held-out cases
            (best, median and worst by WT Dice).

Label names follow the BraTS-PEDs paper (arXiv 2404.15009): 1 ET, 2 NET, 3 CC, 4 ED.
"""
import base64
import csv
import io
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

NN = "/hickory/users/j/o/johnnyta/nnunet"
PKG = "/hickory/users/j/o/johnnyta/PKG - BraTS-PEDs-v1/BraTS-PEDs-v1/Training"
OUT = "/nas/longleaf/home/johnnyta/wanda/2026-11-16_1615_ok-so-for-my-machine-learning-model-i/output"

SETS = {
    "Held-out test set (51 cases, scored once)": {
        "gt": f"{NN}/holdout_test/labels",
        "pred": f"{NN}/scoring/holdout_502_pred",
        "summary": f"{NN}/scoring/holdout_502_pred/summary.json",
        "regions": f"{NN}/scoring/holdout_502/region_summary.json",
        "cases": f"{NN}/scoring/holdout_502/region_cases.csv",
    },
    "Out-of-fold development set (204 cases)": {
        "gt": f"{NN}/nnUNet_raw/Dataset502_BraTSPED_dev/labelsTr",
        "pred": f"{NN}/scoring/oof_502_nnunet",
        "summary": f"{NN}/scoring/oof_502_nnunet/summary.json",
        "regions": f"{NN}/scoring/oof_502/region_summary.json",
        "cases": f"{NN}/scoring/oof_502/region_cases.csv",
    },
}
LABEL_NAMES = {
    1: "label 1 (ET)",
    2: "label 2 (NET)",
    3: "label 3 (CC)",
    4: "label 4 (ED)",
}
LABEL_COLOURS = {1: "#e41a1c", 2: "#4daf4a", 3: "#377eb8", 4: "#ffd92f"}


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def iqr_str(s):
    if s["median"] is None:
        return "NA"
    return f"{s['median']:.3f} ({s['q1']:.3f} to {s['q3']:.3f})"


def label_dice(summary_path):
    d = json.load(open(summary_path))
    vals = {l: [] for l in LABEL_NAMES}
    for case in d["metric_per_case"]:
        for l in LABEL_NAMES:
            m = case["metrics"].get(str(l))
            if m is None:
                continue
            v = m.get("Dice")
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            vals[l].append(float(v))
    return vals


def summarise(v):
    if not v:
        return {"n": 0, "median": None, "q1": None, "q3": None, "mean": None, "zero": 0}
    a = np.array(v)
    return {"n": len(a), "median": float(np.median(a)), "q1": float(np.percentile(a, 25)),
            "q3": float(np.percentile(a, 75)), "mean": float(a.mean()), "zero": int((a == 0).sum())}


def region_boxplot(region_data):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = list(next(iter(region_data.values()))["dice"].keys())
    for ax, metric, ylabel in [(axes[0], "dice", "Dice"), (axes[1], "hd95", "HD95 (mm)")]:
        data, labels = [], []
        for setname, reg in region_data.items():
            for r in ["WT", "TC", "ET"]:
                if metric == "dice":
                    vals = reg["cases"][r]["dice"]
                else:
                    vals = reg["cases"][r]["hd95"]
                data.append(vals)
                labels.append(f"{r}\n{'test' if 'Held' in setname else 'OOF'}")
        ax.boxplot(data, widths=0.6)
        ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        if metric == "dice":
            ax.set_ylim(-0.02, 1.02)
    fig.suptitle("Analysis 1: region accuracy against expert labels (WT, TC, ET)", fontsize=10)
    return fig_to_b64(fig)


def label_boxplot(label_data):
    fig, axes = plt.subplots(1, len(label_data), figsize=(11, 4), sharey=True)
    for ax, (setname, vals) in zip(axes, label_data.items()):
        keys = [l for l in LABEL_NAMES if vals[l]]
        data = [vals[l] for l in keys]
        ax.boxplot(data, widths=0.6)
        for i, l in enumerate(keys, start=1):
            x = np.random.default_rng(l).normal(i, 0.05, size=len(vals[l]))
            ax.scatter(x, vals[l], s=8, alpha=0.5, color=LABEL_COLOURS[l])
            s = summarise(vals[l])
            ax.text(i, 1.03, f"n={s['n']}\n0s={s['zero']}", ha="center", fontsize=7)
        ax.set_xticks(range(1, len(keys) + 1), [LABEL_NAMES[l].replace(" (", "\n(") for l in keys], fontsize=7)
        ax.set_title(setname, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Dice (cases where label is present)")
    axes[0].set_ylim(-0.02, 1.12)
    fig.suptitle("Analysis 2: per-label Dice (n = cases with label in reference; 0s = Dice of zero)", fontsize=10)
    return fig_to_b64(fig)


def load(p):
    return np.asanyarray(nib.load(p).dataobj)


def overlay_figure(cases):
    fig, axes = plt.subplots(len(cases), 2, figsize=(9, 4.4 * len(cases)))
    for r, (tag, cid, dice) in enumerate(cases):
        gt = load(f"{NN}/holdout_test/labels/{cid}.nii.gz").astype(int)
        pr = load(f"{NN}/scoring/holdout_502_pred/{cid}.nii.gz").astype(int)
        flair = load(f"{PKG}/{cid}/{cid}-t2f.nii.gz").astype(float)
        t1c = load(f"{PKG}/{cid}/{cid}-t1c.nii.gz").astype(float)
        z = int(np.argmax((gt > 0).sum(axis=(0, 1))))
        for c, (img, name) in enumerate([(flair, "T2-FLAIR"), (t1c, "T1 post-contrast")]):
            ax = axes[r, c]
            sl = np.rot90(img[:, :, z])
            lo, hi = np.percentile(img[img > 0], [1, 99.5]) if (img > 0).any() else (0, 1)
            ax.imshow(sl, cmap="gray", vmin=lo, vmax=hi)
            for l, col in LABEL_COLOURS.items():
                g = np.rot90((gt[:, :, z] == l).astype(float))
                p = np.rot90((pr[:, :, z] == l).astype(float))
                if g.any():
                    ax.contour(g, levels=[0.5], colors=[col], linewidths=1.5, linestyles="solid")
                if p.any():
                    ax.contour(p, levels=[0.5], colors=[col], linewidths=1.2, linestyles="dashed")
            ax.set_title(f"{tag}: {cid}, WT Dice {dice:.3f}\n{name}, axial slice {z}", fontsize=8)
            ax.axis("off")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, lw=2, label=LABEL_NAMES[l]) for l, c in LABEL_COLOURS.items()]
    handles += [Line2D([0], [0], color="k", lw=1.5, ls="solid", label="expert (solid)"),
                Line2D([0], [0], color="k", lw=1.5, ls="dashed", label="model (dashed)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, frameon=False)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    return fig_to_b64(fig)


def pick_cases(cases_csv, n=3):
    wt = []
    for row in csv.DictReader(open(cases_csv)):
        if row["region"] == "WT" and row["status"] == "ok":
            wt.append((float(row["dice"]), row["case"]))
    wt.sort()
    idx = [0, len(wt) // 2, len(wt) - 1]
    return [("worst", wt[0][1], wt[0][0]), ("median", wt[len(wt) // 2][1], wt[len(wt) // 2][0]),
            ("best", wt[-1][1], wt[-1][0])]


def main():
    os.makedirs(OUT, exist_ok=True)
    region_data, label_data, tables = {}, {}, []
    for setname, paths in SETS.items():
        regs = json.load(open(paths["regions"]))["regions"]
        cases = {r: {"dice": [], "hd95": []} for r in ["WT", "TC", "ET"]}
        for row in csv.DictReader(open(paths["cases"])):
            if row["status"] != "both_empty":
                cases[row["region"]]["dice"].append(float(row["dice"]))
            if row["hd95_mm"] not in ("", "nan"):
                cases[row["region"]]["hd95"].append(float(row["hd95_mm"]))
        region_data[setname] = {"regs": regs, "cases": cases, "dice": {r: cases[r]["dice"] for r in cases}}
        label_data[setname] = label_dice(paths["summary"])
        region_data[setname]["dice"] = {r: cases[r]["dice"] for r in cases}

    region_img = region_boxplot(region_data)
    label_img = label_boxplot(label_data)
    picks = pick_cases(SETS["Held-out test set (51 cases, scored once)"]["cases"])
    overlay_img = overlay_figure(picks)

    for img, name in [(region_img, "analysis1_region_boxplots.png"),
                      (label_img, "analysis2_label_boxplots.png"),
                      (overlay_img, "analysis3_overlays_test.png")]:
        with open(os.path.join(OUT, name), "wb") as fh:
            fh.write(base64.b64decode(img))

    reg_rows = []
    for setname, d in region_data.items():
        for r in ["WT", "TC", "ET"]:
            s = d["regs"][r]
            reg_rows.append(
                f"<tr><td>{setname}</td><td>{r}</td><td>{s['labels']}</td>"
                f"<td>{iqr_str(s['dice'])}</td><td>{s['dice']['mean']:.3f}</td><td>{s['dice']['n']}</td>"
                f"<td>{iqr_str(s['hd95_mm'])}</td><td>{s['both_empty_excluded']}</td>"
                f"<td>{s['one_empty_dice_zero']}</td></tr>")
    label_rows = []
    for setname, vals in label_data.items():
        for l in LABEL_NAMES:
            s = summarise(vals[l])
            label_rows.append(
                f"<tr><td>{setname}</td><td>{LABEL_NAMES[l]}</td><td>{s['n']}</td>"
                f"<td>{iqr_str(s)}</td><td>{s['mean']:.3f}</td><td>{s['zero']}</td></tr>"
                if s["n"] else
                f"<tr><td>{setname}</td><td>{LABEL_NAMES[l]}</td><td>0</td><td>NA</td><td>NA</td><td>0</td></tr>")

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>BraTS-PEDs nnU-Net results</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;color:#222;max-width:1000px}}
table{{border-collapse:collapse;margin:10px 0 18px}}td,th{{border:1px solid #bbb;padding:5px 9px;font-size:13px}}
th{{background:#f0f0f0}}.warn{{background:#fff4d6;border:1px solid #e0b040;padding:10px 14px;margin:14px 0}}
small{{color:#555}}h2{{margin-top:28px}}img{{max-width:100%}}</style></head><body>
<h1>BraTS-PEDs: nnU-Net segmentation results</h1>
<p>Model: nnU-Net 3d_fullres, default trainer, 1000 epochs per fold, 5-fold ensemble for the test set.
Reference: BraTS-PEDs expert annotations. Training pool: 255 cases (two skull-stripped cases excluded).
Development: 204 cases. Held-out test: 51 cases, scored once.</p>
<div class="warn"><b>Label names</b> follow the BraTS-PEDs paper (arXiv 2404.15009): 1 = ET, 2 = NET, 3 = CC, 4 = ED. TC = labels 1, 2, 3 (ET + NET + CC). WT = labels 1-4. ET = label 1.</div>

<h2>Analysis 1: Segmentation accuracy against expert annotations (Dice, HD95)</h2>
<img src="data:image/png;base64,{region_img}" alt="Region boxplots">
<table><tr><th>Set</th><th>Region</th><th>Labels</th><th>Dice median (IQR)</th><th>Dice mean</th><th>n</th>
<th>HD95 mm median (IQR)</th><th>Both empty (excluded)</th><th>One empty (Dice 0)</th></tr>
{''.join(reg_rows)}</table>
<small>Dice is computed per case. Where the region is empty in both reference and prediction, Dice is undefined and the case is excluded.
Where it is empty in only one, Dice is 0. The held-out test set is the final estimate. The development set is internal validation.</small>

<h2>Analysis 2: Subregion performance (per label)</h2>
<img src="data:image/png;base64,{label_img}" alt="Label boxplots">
<table><tr><th>Set</th><th>Label</th><th>n (label present)</th><th>Dice median (IQR)</th><th>Dice mean</th><th>Dice = 0</th></tr>
{''.join(label_rows)}</table>
<small>Per-label Dice is taken from nnU-Net's own per-label output and is computed only where the label is present in the reference.
Dice = 0 cases are where the label is present but missed, or predicted where absent.</small>

<h2>Analysis 3: Visual comparison (held-out test cases)</h2>
<p>Three held-out cases chosen by WT Dice: worst, median and best. Solid outlines are expert annotations, dashed outlines are the model.
Each row shows one case on T2-FLAIR and T1 post-contrast, at the axial slice with the largest reference tumour. Subject to change.</p>
<img src="data:image/png;base64,{overlay_img}" alt="Overlays">
</body></html>"""
    with open(os.path.join(OUT, "results_visualization.html"), "w") as fh:
        fh.write(html)
    print("written:", os.path.join(OUT, "results_visualization.html"))
    print("cases shown in analysis 3:", picks)


if __name__ == "__main__":
    main()
