#!/usr/bin/env python3
"""Per-subject checks for skull-stripped BraTS-PEDs data.

Checks, for every subject in Training and Validation:
  - geometry: all modalities, brainmask and seg share shape and affine
  - background: image voxels that are nonzero outside the brainmask
                (should be 0 if stripping zeroed the background)
  - holes: voxels inside the brainmask that are exactly zero, per modality
  - seg: label voxels that fall outside the brainmask (tumour removed by stripping)
Writes one CSV row per subject and prints a summary.
"""
import csv
import glob
import os
import sys
from collections import Counter

import nibabel as nib
import numpy as np

ROOT = sys.argv[1]
OUT_CSV = sys.argv[2]
MODS = ["t1n", "t1c", "t2w", "t2f"]

rows = []
for split in ["Training", "Validation"]:
    for d in sorted(glob.glob(os.path.join(ROOT, split, "*"))):
        if not os.path.isdir(d):
            continue
        s = os.path.basename(d)

        def p(suffix):
            return os.path.join(d, f"{s}-{suffix}.nii.gz")

        bm_img = nib.load(p("brainmask"))
        bm_arr = np.asanyarray(bm_img.dataobj)
        bm = bm_arr > 0
        ref_shape, ref_aff = bm_img.shape, bm_img.affine

        row = {
            "subject": s,
            "split": split,
            "shape": "x".join(map(str, ref_shape)),
            "brainmask_voxels": int(bm.sum()),
        }
        geom_ok = True

        for m in MODS:
            im = nib.load(p(m))
            if im.shape != ref_shape or not np.allclose(im.affine, ref_aff, atol=1e-3):
                geom_ok = False
            arr = np.asanyarray(im.dataobj)
            nz = arr != 0
            row[f"{m}_nonzero_outside_mask"] = int((nz & ~bm).sum())
            row[f"{m}_zero_inside_mask_frac"] = round(float((~nz & bm).sum()) / max(int(bm.sum()), 1), 5)

        seg_path = p("seg")
        if os.path.exists(seg_path):
            seg_img = nib.load(seg_path)
            seg_arr = np.asanyarray(seg_img.dataobj)
            if seg_img.shape != ref_shape or not np.allclose(seg_img.affine, ref_aff, atol=1e-3):
                geom_ok = False
            seg = seg_arr > 0
            row["seg_voxels"] = int(seg.sum())
            row["seg_outside_mask"] = int((seg & ~bm).sum())
            row["seg_labels"] = ",".join(str(int(v)) for v in np.unique(seg_arr))
        else:
            row["seg_voxels"] = ""
            row["seg_outside_mask"] = ""
            row["seg_labels"] = ""

        row["geometry_ok"] = geom_ok
        rows.append(row)
        print(s, flush=True)

fieldnames = list(rows[0].keys())
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print("\n=== SUMMARY ===")
for split in ["Training", "Validation"]:
    rs = [r for r in rows if r["split"] == split]
    print(f"\n{split}: {len(rs)} subjects")
    print("  geometry mismatch:", sum(not r["geometry_ok"] for r in rs))
    print("  shapes:", dict(Counter(r["shape"] for r in rs)))
    for m in MODS:
        n_bad = sum(r[f"{m}_nonzero_outside_mask"] > 0 for r in rs)
        print(f"  {m}: subjects with nonzero voxels outside brainmask: {n_bad}")
        fr = [r[f"{m}_zero_inside_mask_frac"] for r in rs]
        print(f"  {m}: zero-inside-mask fraction median {np.median(fr):.5f}, max {np.max(fr):.5f}")
    seg_rs = [r for r in rs if r["seg_voxels"] != ""]
    if seg_rs:
        bad = [r for r in seg_rs if r["seg_outside_mask"] > 0]
        print(f"  seg: subjects with label voxels outside brainmask: {len(bad)} of {len(seg_rs)}")
        if bad:
            tot = sum(r["seg_outside_mask"] for r in bad)
            print(f"  seg: total label voxels outside brainmask: {tot}")
            for r in sorted(bad, key=lambda r: -r["seg_outside_mask"])[:10]:
                print(f"    {r['subject']}: {r['seg_outside_mask']} of {r['seg_voxels']} label voxels outside mask")
        print("  seg label values seen:", dict(Counter(r["seg_labels"] for r in seg_rs)))
