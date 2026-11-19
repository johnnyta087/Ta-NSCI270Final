#!/usr/bin/env python3
"""Region-level Dice and HD95 for BraTS-PEDs predictions.

Regions (integer labels as in dataset.json):
  WT = labels 1,2,3,4 (whole tumour)
  TC = labels 1,2,3   (ET + NET + CC, standard BraTS definition)
  ET = label 1        (enhancing tumour)

Usage: score_regions.py GT_DIR PRED_DIR OUT_DIR TITLE
Writes region_cases.csv, region_summary.json and region_report.html into OUT_DIR.
Cases where a region is empty in both GT and prediction are excluded from Dice (counted).
Cases where it is empty in only one get Dice 0 and no HD95 (counted).
"""
import base64
import csv
import io
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

REGIONS = {"WT": [1, 2, 3, 4], "TC": [1, 2, 3], "ET": [1]}


def dice(a, b):
    return 2.0 * np.logical_and(a, b).sum() / (a.sum() + b.sum())


def surface(m):
    return np.logical_and(m, np.logical_not(ndi.binary_erosion(m)))


def hd95(a, b, spacing):
    sa, sb = surface(a), surface(b)
    da = ndi.distance_transform_edt(np.logical_not(sb), sampling=spacing)
    db = ndi.distance_transform_edt(np.logical_not(sa), sampling=spacing)
    d = np.concatenate([da[sa], db[sb]])
    return float(np.percentile(d, 95))


def stats(vals):
    v = np.array([x for x in vals if x is not None and not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return {"n": 0, "median": None, "q1": None, "q3": None, "mean": None}
    return {"n": int(len(v)), "median": float(np.median(v)),
            "q1": float(np.percentile(v, 25)), "q3": float(np.percentile(v, 75)),
            "mean": float(v.mean())}


def fmt(x):
    return "NA" if x is None else f"{x:.3f}"


def main():
    gt_dir, pred_dir, out_dir, title = sys.argv[1:5]
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(gt_dir) if f.endswith(".nii.gz"))
    rows, missing = [], []
    for f in files:
        pp = os.path.join(pred_dir, f)
        if not os.path.exists(pp):
            missing.append(f)
            continue
        gi = nib.load(os.path.join(gt_dir, f))
        g = np.asanyarray(gi.dataobj).astype(np.int16)
        p = np.asanyarray(nib.load(pp).dataobj).astype(np.int16)
        if p.shape != g.shape:
            raise SystemExit(f"shape mismatch for {f}: {p.shape} vs {g.shape}")
        spacing = gi.header.get_zooms()[:3]
        for reg, labs in REGIONS.items():
            gm = np.isin(g, labs)
            pm = np.isin(p, labs)
            if not gm.any() and not pm.any():
                status, d, h = "both_empty", np.nan, np.nan
            elif not gm.any() or not pm.any():
                status, d, h = "one_empty", 0.0, np.nan
            else:
                status, d, h = "ok", dice(gm, pm), hd95(gm, pm, spacing)
            rows.append({"case": f[:-7], "region": reg, "dice": d, "hd95_mm": h,
                         "status": status, "gt_voxels": int(gm.sum()),
                         "pred_voxels": int(pm.sum())})

    with open(os.path.join(out_dir, "region_cases.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {"title": title, "cases_scored": len(files) - len(missing),
               "cases_missing_prediction": missing, "regions": {}}
    for reg in REGIONS:
        r = [x for x in rows if x["region"] == reg]
        summary["regions"][reg] = {
            "labels": REGIONS[reg],
            "dice": stats([x["dice"] for x in r if x["status"] != "both_empty"]),
            "hd95_mm": stats([x["hd95_mm"] for x in r]),
            "both_empty_excluded": sum(x["status"] == "both_empty" for x in r),
            "one_empty_dice_zero": sum(x["status"] == "one_empty" for x in r),
        }
    with open(os.path.join(out_dir, "region_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # Figure: Dice and HD95 per region
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    dice_data = [[x["dice"] for x in rows if x["region"] == r and x["status"] != "both_empty"] for r in REGIONS]
    hd_data = [[x["hd95_mm"] for x in rows if x["region"] == r and not np.isnan(x["hd95_mm"])] for r in REGIONS]
    axes[0].boxplot(dice_data)
    axes[0].set_xticks(range(1, len(REGIONS) + 1), list(REGIONS))
    axes[0].set_ylabel("Dice")
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].boxplot(hd_data)
    axes[1].set_xticks(range(1, len(REGIONS) + 1), list(REGIONS))
    axes[1].set_ylabel("HD95 (mm)")
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    img = base64.b64encode(buf.getvalue()).decode()
    with open(os.path.join(out_dir, "region_boxplots.png"), "wb") as fh:
        fh.write(base64.b64decode(img))

    # HTML report
    trs = []
    for reg in REGIONS:
        s = summary["regions"][reg]
        d, h = s["dice"], s["hd95_mm"]
        trs.append(
            f"<tr><td>{reg}</td><td>{REGIONS[reg]}</td>"
            f"<td>{fmt(d['median'])} ({fmt(d['q1'])} to {fmt(d['q3'])})</td>"
            f"<td>{fmt(d['mean'])}</td><td>{d['n']}</td>"
            f"<td>{fmt(h['median'])} ({fmt(h['q1'])} to {fmt(h['q3'])})</td>"
            f"<td>{h['n']}</td><td>{s['both_empty_excluded']}</td><td>{s['one_empty_dice_zero']}</td></tr>")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;color:#222}}
table{{border-collapse:collapse;margin:12px 0}}td,th{{border:1px solid #bbb;padding:6px 10px;font-size:14px}}
th{{background:#f0f0f0}}small{{color:#555}}</style></head><body>
<h2>{title}</h2>
<p>Cases scored: {summary['cases_scored']}. Missing predictions: {len(missing)}.</p>
<table><tr><th>Region</th><th>Labels</th><th>Dice median (IQR)</th><th>Dice mean</th><th>n</th>
<th>HD95 mm median (IQR)</th><th>n</th><th>Both empty (excluded)</th><th>One empty (Dice 0)</th></tr>
{''.join(trs)}</table>
<img src="data:image/png;base64,{img}" alt="Dice and HD95 per region" style="max-width:100%">
<p><small>WT = labels 1-4; TC = labels 1, 2, 3; ET = label 1. Per-case values: region_cases.csv.
HD95 is computed in millimetres from the voxel spacing and is only defined when both masks are non-empty.</small></p>
</body></html>"""
    with open(os.path.join(out_dir, "region_report.html"), "w") as fh:
        fh.write(html)

    for reg in REGIONS:
        s = summary["regions"][reg]
        print(f"{reg}: Dice median {fmt(s['dice']['median'])} (IQR {fmt(s['dice']['q1'])}-{fmt(s['dice']['q3'])}), "
              f"n={s['dice']['n']}; HD95 median {fmt(s['hd95_mm']['median'])} mm, n={s['hd95_mm']['n']}")
    print("missing predictions:", len(missing))


if __name__ == "__main__":
    main()
