#!/usr/bin/env python3
"""Create a seeded, stratified held-out test split and the nnU-Net dataset folders.

Pool: BraTS-PEDs Training cases minus the two skull-stripped cases (00024, 00098) = 255.
Strata: source x survival availability (CBTN+survival, CBTN-no-survival, DIPGr, DFCI).
Test: 51 cases (~20%), allocated to strata proportionally (largest-remainder rounding),
      chosen at random within each stratum with a fixed seed. Never used for training.
Dev:  remaining 204 cases -> nnUNet_raw/Dataset502_BraTSPED_dev (links only).
Holdout: the 51 test cases -> holdout_test/ (links only, outside nnUNet_raw).
Nothing is copied: all image and label files are symbolic links into the PKG folder.
"""
import csv
import json
import math
import os
import random
import sys
from collections import OrderedDict

SEED = 12345
TEST_N = 51
META = "/hickory/users/j/o/johnnyta/brats-peds-stripped/BraTS-PEDs_metadata (1).tsv"
PKG = "/hickory/users/j/o/johnnyta/PKG - BraTS-PEDs-v1/BraTS-PEDs-v1/Training"
NNRAW = "/hickory/users/j/o/johnnyta/nnunet/nnUNet_raw"
DSNAME = "Dataset502_BraTSPED_dev"
HOLD = "/hickory/users/j/o/johnnyta/nnunet/holdout_test"
OUT = "/nas/longleaf/home/johnnyta/wanda/2026-11-16_1615_ok-so-for-my-machine-learning-model-i/output"
EXCL = {"BraTS-PED-00024-000", "BraTS-PED-00098-000"}
MODS = ["t1n", "t1c", "t2w", "t2f"]  # channel 0..3, same order as Dataset501
LABELS = {"background": 0, "ET": 1, "NET": 2, "CC": 3, "ED": 4}
CHANNELS = {"0": "T1n", "1": "T1c", "2": "T2w", "3": "T2-FLAIR"}


def has_survival(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def main():
    if os.path.exists(os.path.join(NNRAW, DSNAME)) or os.path.exists(HOLD):
        sys.exit("Refusing to overwrite: dataset or holdout folder already exists.")

    rows = list(csv.DictReader(open(META, encoding="utf-8"), delimiter="\t"))
    train_rows = [r for r in rows if r["BraTS2025_cohort"] == "Training"]
    train_ids = {r["BraTS-SubjectID"] for r in train_rows}
    for cid in EXCL:
        if cid not in train_ids:
            sys.exit(f"Excluded case not found in Training metadata: {cid}")

    cases = {}
    for r in train_rows:
        cid = r["BraTS-SubjectID"]
        if cid in EXCL:
            continue
        d = os.path.join(PKG, cid)
        files = [os.path.join(d, f"{cid}-{m}.nii.gz") for m in MODS] + [os.path.join(d, f"{cid}-seg.nii.gz")]
        for f in files:
            if not os.path.exists(f):
                sys.exit(f"Missing file: {f}")
        surv = has_survival(r["Overall survival (days)"])
        cases[cid] = {
            "source": r["Source"],
            "has_survival": surv,
            "os_days": r["Overall survival (days)"].strip() if surv else "",
        }
    if len(cases) != 255:
        sys.exit(f"Expected 255 pool cases, found {len(cases)}")

    # Strata: source x survival availability
    strata = OrderedDict()
    for cid in sorted(cases):
        c = cases[cid]
        key = f"{c['source']}|{'survival' if c['has_survival'] else 'no_survival'}"
        strata.setdefault(key, []).append(cid)
    pool_n = len(cases)
    quota = {k: TEST_N * len(v) / pool_n for k, v in strata.items()}
    alloc = {k: int(math.floor(quota[k])) for k in strata}
    remainder = TEST_N - sum(alloc.values())
    for k in sorted(strata, key=lambda k: (-(quota[k] - alloc[k]), k))[:remainder]:
        alloc[k] += 1
    assert sum(alloc.values()) == TEST_N

    rng = random.Random(SEED)
    test_ids = []
    for k in strata:
        ids = sorted(strata[k])
        rng.shuffle(ids)
        test_ids.extend(ids[: alloc[k]])
    test_set = set(test_ids)
    dev_ids = sorted(c for c in cases if c not in test_set)
    assert len(test_set) == TEST_N and len(dev_ids) == pool_n - TEST_N
    assert not (test_set & set(dev_ids))

    # Split CSV and summary
    os.makedirs(OUT, exist_ok=True)
    csv_path = os.path.join(OUT, "brats_ped_split_seed12345.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "source", "has_survival", "os_days", "stratum", "split", "seed"])
        for cid in sorted(cases):
            c = cases[cid]
            key = f"{c['source']}|{'survival' if c['has_survival'] else 'no_survival'}"
            w.writerow([cid, c["source"], int(c["has_survival"]), c["os_days"], key,
                        "test" if cid in test_set else "dev", SEED])

    counts = {}
    for k in strata:
        counts[k] = {"pool": len(strata[k]), "test": alloc[k], "dev": len(strata[k]) - alloc[k]}
    summary = {
        "seed": SEED,
        "pool_n": pool_n,
        "excluded": sorted(EXCL),
        "test_n": TEST_N,
        "dev_n": len(dev_ids),
        "test_survival_n": sum(1 for c in test_ids if cases[c]["has_survival"]),
        "dev_survival_n": sum(1 for c in dev_ids if cases[c]["has_survival"]),
        "strata": counts,
        "dataset_dev": DSNAME,
        "holdout_dir": HOLD,
    }
    with open(os.path.join(OUT, "brats_ped_split_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # nnU-Net development dataset (links only)
    ds = os.path.join(NNRAW, DSNAME)
    os.makedirs(os.path.join(ds, "imagesTr"))
    os.makedirs(os.path.join(ds, "labelsTr"))
    os.makedirs(os.path.join(ds, "imagesTs"))  # empty: no unlabelled cases used
    for cid in dev_ids:
        d = os.path.join(PKG, cid)
        for i, m in enumerate(MODS):
            os.symlink(os.path.join(d, f"{cid}-{m}.nii.gz"),
                       os.path.join(ds, "imagesTr", f"{cid}_000{i}.nii.gz"))
        os.symlink(os.path.join(d, f"{cid}-seg.nii.gz"),
                   os.path.join(ds, "labelsTr", f"{cid}.nii.gz"))
    dataset_json = {
        "channel_names": CHANNELS,
        "labels": LABELS,
        "numTraining": len(dev_ids),
        "file_ending": ".nii.gz",
    }
    with open(os.path.join(ds, "dataset.json"), "w", encoding="utf-8") as fh:
        json.dump(dataset_json, fh, indent=4)

    # Holdout test folder (links only, outside nnUNet_raw)
    os.makedirs(os.path.join(HOLD, "images"))
    os.makedirs(os.path.join(HOLD, "labels"))
    for cid in sorted(test_ids):
        d = os.path.join(PKG, cid)
        for i, m in enumerate(MODS):
            os.symlink(os.path.join(d, f"{cid}-{m}.nii.gz"),
                       os.path.join(HOLD, "images", f"{cid}_000{i}.nii.gz"))
        os.symlink(os.path.join(d, f"{cid}-seg.nii.gz"),
                   os.path.join(HOLD, "labels", f"{cid}.nii.gz"))
    with open(os.path.join(HOLD, "test_cases.csv"), "w", encoding="utf-8") as fh:
        fh.write("case_id\n")
        for cid in sorted(test_ids):
            fh.write(cid + "\n")

    # Checks: counts, every link resolves, no overlap between dev and test
    assert len(os.listdir(os.path.join(ds, "imagesTr"))) == 4 * len(dev_ids)
    assert len(os.listdir(os.path.join(ds, "labelsTr"))) == len(dev_ids)
    assert len(os.listdir(os.path.join(HOLD, "images"))) == 4 * TEST_N
    assert len(os.listdir(os.path.join(HOLD, "labels"))) == TEST_N
    for root in (os.path.join(ds, "imagesTr"), os.path.join(ds, "labelsTr"),
                 os.path.join(HOLD, "images"), os.path.join(HOLD, "labels")):
        for f in os.listdir(root):
            assert os.path.exists(os.path.join(root, f)), f"broken link {f}"
    dev_case_ids = {f.split("_000")[0].replace(".nii.gz", "") for f in os.listdir(os.path.join(ds, "labelsTr"))}
    assert not (dev_case_ids & test_set), "test cases leaked into development folder"

    print(json.dumps(summary, indent=2))
    print("split CSV:", csv_path)


if __name__ == "__main__":
    main()
