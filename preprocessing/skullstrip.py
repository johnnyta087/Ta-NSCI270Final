#!/usr/bin/env python
"""
Batch skull-strip BraTS-PEDs with HD-BET.

Usage: python skullstrip.py <input_root> <output_root>

For each patient folder:
  1. Run HD-BET on the T1c scan to get a brain mask
  2. Apply that mask to t1n, t1c, t2w, t2f (they are co-registered)
  3. Copy seg unchanged (if present; validation has none)
Output keeps the same folder/file names, so downstream code works as-is.
Resumable: patients with a DONE marker are skipped. Failures go to failed.txt.
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes

IN_ROOT, OUT_ROOT = sys.argv[1], sys.argv[2]
MASK_SEQ = "t1c"
SEQS = ["t1n", "t1c", "t2w", "t2f"]


def find(folder, suffix):
    m = glob.glob(os.path.join(folder, f"*{suffix}.nii.gz"))
    return m[0] if m else None


def process(pt_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    src = find(pt_dir, MASK_SEQ)
    if src is None:
        raise FileNotFoundError(f"no {MASK_SEQ} in {pt_dir}")

    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        stripped_path = os.path.join(tmp, "stripped.nii.gz")
        subprocess.run(
            ["hd-bet", "-i", src, "-o", stripped_path, "-device", "cuda"],
            check=True,
        )
        # HD-BET zeros everything outside the brain, so the mask is nonzero voxels.
        # (Version-proof: avoids depending on HD-BET's mask file naming.)
        outs = glob.glob(os.path.join(tmp, "*.nii.gz"))
        if not outs:
            raise FileNotFoundError("HD-BET produced no output file")
        stripped = nib.load(outs[0]).get_fdata()
        mask = binary_fill_holes(stripped != 0)

    ref = nib.load(src)
    for s in SEQS:
        f = find(pt_dir, s)
        if f is None:
            continue
        img = nib.load(f)
        data = img.get_fdata()
        if data.shape != mask.shape:
            raise ValueError(f"shape mismatch {s}: {data.shape} vs {mask.shape}")
        out = (data * mask).astype(img.get_data_dtype())
        nib.save(nib.Nifti1Image(out, img.affine, img.header),
                 os.path.join(out_dir, os.path.basename(f)))

    seg = find(pt_dir, "seg")
    if seg:
        shutil.copy(seg, os.path.join(out_dir, os.path.basename(seg)))

    base = os.path.basename(src).replace(f"{MASK_SEQ}.nii.gz", "brainmask.nii.gz")
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), ref.affine, ref.header),
             os.path.join(out_dir, base))

    open(os.path.join(out_dir, "DONE"), "w").close()


def main():
    patients = sorted(p for p in os.listdir(IN_ROOT)
                      if os.path.isdir(os.path.join(IN_ROOT, p)))
    print(f"{len(patients)} patients in {IN_ROOT}", flush=True)
    os.makedirs(OUT_ROOT, exist_ok=True)
    failed_log = os.path.join(OUT_ROOT, "failed.txt")

    for i, p in enumerate(patients, 1):
        out_dir = os.path.join(OUT_ROOT, p)
        if os.path.exists(os.path.join(out_dir, "DONE")):
            print(f"[{i}/{len(patients)}] {p} already done, skipping", flush=True)
            continue
        try:
            process(os.path.join(IN_ROOT, p), out_dir)
            print(f"[{i}/{len(patients)}] {p} ok", flush=True)
        except Exception:
            print(f"[{i}/{len(patients)}] {p} FAILED", flush=True)
            traceback.print_exc()
            with open(failed_log, "a") as fh:
                fh.write(p + "\n")


if __name__ == "__main__":
    main()
