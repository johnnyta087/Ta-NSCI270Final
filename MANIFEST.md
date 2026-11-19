# Code manifest: BraTS-PEDs nnU-Net segmentation, scoring and figures

Copies of the scripts used for this project. Paths inside the scripts point to the Longleaf locations
where they ran (Hickory project folder and Wanda output folder).

## 1. Split (run first)
| File | What it does |
|---|---|
| `split/make_split_502.py` | Excludes BraTS-PED-00024 and 00098 (skull-stripped), then makes a seeded (12345), source-stratified split: 51 held-out test cases and 204 development cases. Builds the nnU-Net dataset `Dataset502_BraTSPED_dev` and the holdout folder from symbolic links only. |

The split itself is in the output folder: `brats_ped_split_seed12345.csv` (one row per case with source, survival availability and split) and `brats_ped_split_summary.json`.

## 2. Preprocessing
| File | What it does |
|---|---|
| `preprocessing/skullstrip.py`, `skullstrip.sl`, `skullstrip_val.sl` | HD-BET skull stripping of the training and validation sets. Used for the stripped copies, which are not used for training. |
| `preprocessing/check_masks.py` | QC of skull-stripped masks (geometry, background, holes, tumour outside the mask). |

## 3. Training (nnU-Net v2, 3d_fullres, default trainer)
| File | What it does |
|---|---|
| `slurm/plan_501.sbatch`, `slurm/smoke_501.sbatch` | First attempt on Dataset501 (255 cases): planning and a 5-epoch smoke test. Not used for the final results. |
| `slurm/plan_502.sbatch` | Planning and preprocessing for Dataset502 (204 development cases). |
| `slurm/poc_502_fold0.sbatch` | Proof-of-concept, fold 0, 100 epochs. Not used for the final results. |
| `slurm/cv_502_array.sbatch` | Five-fold array job, 1000 epochs per fold. Folds 0 and 2 hit the 16-hour limit. |
| `slurm/resume_502.sbatch` | Resumes an interrupted fold from its latest checkpoint (`--c`). Run for folds 0 and 2 (`FOLD` variable). |

The nnU-Net training used the default trainer; no custom training code was written.

## 4. Scoring
| File | What it does |
|---|---|
| `slurm/score_oof_502.sbatch` | Combines the five folds' out-of-fold predictions (`nnUNetv2_accumulate_crossval_results`) and scores the 204 development cases. |
| `slurm/score_holdout_502.sbatch` | Runs the five-fold ensemble once on the 51 held-out cases (`nnUNetv2_predict`), then nnU-Net per-label evaluation (`nnUNetv2_evaluate_folder`). |
| `tools/score_regions.py` | Region scores from label groups: WT = labels 1-4, TC = labels 1-3, ET = label 1. Dice and HD95 per case, with summary JSON and report. |
| `slurm/rescore_502.sbatch` | First region-scoring rerun, with the old TC definition (labels 1, 3, 4). Superseded. |
| `slurm/rescore_viz_502.sbatch` | Current region rescoring (TC = labels 1-3, paper label names) and report rebuild. |

## 5. Visualisation
| File | What it does |
|---|---|
| `tools/build_results_viz.py` | Builds `results_visualization.html` (all three analyses) and the combined figures. Holds the label names (1 ET, 2 NET, 3 CC, 4 ED) and colours. |
| `slurm/viz_502.sbatch` | Runs `build_results_viz.py`. |
| `tools/export_pngs.py`, `slurm/export_png_502.sbatch` | Separate untitled PNG for each graph (`png_exports/`). |
| `tools/combine_pngs.py`, `slurm/combine_png_502.sbatch` | Three combined PNGs with legends: `analysis1_region.png`, `analysis2_labels.png`, `analysis3_overlays.png`. |

## Not saved as files
- The dataset-conversion step for Dataset501 was run before this chat, and I don't have a copy of that script in the project folder. The Dataset501 `dataset.json` and links are in `nnunet/nnUNet_raw/Dataset501_BraTSPED`.
- One-off checks were run as inline commands and are not saved: the label intensity and layout checks, the label-mapping overlay (`label_mapping_check.png`) and the survival-metadata summaries. Ask if you want any of these saved as scripts.

## Run order
1. `split/make_split_502.py`
2. `slurm/plan_502.sbatch`
3. `slurm/cv_502_array.sbatch` (then `resume_502.sbatch` for any fold that times out)
4. `slurm/score_oof_502.sbatch` and `slurm/score_holdout_502.sbatch` (scoring)
5. `slurm/rescore_viz_502.sbatch` (region scores and report)
6. `slurm/export_png_502.sbatch`, then `slurm/combine_png_502.sbatch` (figures)

Survival analysis: not started, no code.
