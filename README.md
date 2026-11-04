# ECE 1512 Project A: ABMIL and a focal-loss extension

This project completes the supplied notebook and compares **ABMIL + cross entropy** with **identical ABMIL + focal loss** (gamma 2, no alpha/class weighting) on CAMELYON16, CAMELYON17 and BRACS. A second, unimplemented proposal investigates coordinate-neighbor smoothing of attention. The literature review covers ABMIL, CLAM, DSMIL, TransMIL and DTFD-MIL.

## Deliverables

- `ECE1512_ProjectA_Student.ipynb`: original implementation cells completed; shared executable modules in `models.py`, `training.py`, `run_experiments.py`.
- `results/<dataset>_<method>_seed42/`: selected checkpoint, config, split manifest, history, test probabilities, measured metrics. Only `metrics.json` with `status: complete` denotes a finished run.
- `output/figures/`: loss/validation curves, confusion matrices, real WSI attention overlays, raw weights/coordinates in NPZ files.
- `output/results_summary.csv`, `output/per_class_results.csv`, `output/paired_changes.csv`: measured tables.
- `output/verification.txt`: software checks, separate from pathology performance.

## Environment

Python 3.12 was used. Create an isolated environment. For an NVIDIA GPU install the appropriate CUDA PyTorch build first (consult the official PyTorch installer); CPU is supported. From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -r requirements.txt
```

The development environment here is at `../.venv/`. Measured-run versions are in each `metrics.json`; complete development dependencies are in `requirements-lock.txt`. GPU torch may require its separate package index when reproducing a lock.

## Course data

Official sources: https://zenodo.org/records/23130543 and https://huggingface.co/datasets/LeonYe/ECE1512-WSI-Feature . Use these pre-extracted features, not arbitrary challenge files.

```text
feature/C16/patch_feats_pretrain_medical_ssl.h5
feature/C17/patch_feats_pretrain_medical_ssl.h5
feature/BRACS/patch_feats_pretrain_medical_ssl.h5
slide/test_067.tif
slide/test_068.tif
slide/test_068.xml
```

## Fixed protocol

| Dataset | Classes | Split |
|---|---:|---|
| C16 | 2 | `test` IDs held out; stratified10% validation from others, seed42 |
| C17 | 4 | Centers>=3 held out;10% validation from other centers, seed42 |
| BRACS | 7 | Supplied CSV splits; no relabeling |

C17 is a four-class slide center-transfer experiment, **not the official patient-stage challenge endpoint**. Supplied train/validation splitting is preserved. Case overlap is audited in split manifests and discussed as a limitation. Test centers are separate.

Both methods: full bags, batch1, seed42, projection256, gated attention128, dropout0, AdamW lr1e-4, weight decay1e-5,30 epochs, constant learning rate. Starter `warmup_epoch` and `n_res` are inactive and not claimed as used.

Checkpoint selection maximizes validation macro one-vs-rest AUROC + macro F1, retaining the earliest tied epoch. Test evaluation occurs after selection. AUC is the mean of class-vs-rest AUCs; missing classes make it undefined. The runner rejects undefined validation AUC. F1 averages all configured classes with zero for undefined precision/recall. Accuracy is percent; AUC/F1 are fractions. Both methods also save ordinary cross entropy at evaluation, because their training objective values use different scales.

One paired seed supports descriptive differences, not statistical significance. Focal loss is attributed to Lin et al.; this is a controlled adaptation, not invention of the loss. It focuses on hard slides, not directly on sparse positive patches, and does not guarantee minority-class improvement.

A GPU storage cache retains source-precision features for complete bags within 55% of initially free GPU memory. Model inputs remain float32, and uncached bags still load lazily. It requires zero DataLoader workers and disables pinning. Twenty tests verify the core pipeline and exact cache equivalence; complete C17 repeats reproduce all 30 epoch metrics and test probabilities exactly. Use `--disable-gpu-cache` to read every bag from HDF5.

## Reproduce

Run the notebook from this directory or set PROJECT_DIR. Model/training functions are exposed directly; the extension cell loads finished run artifacts or trains missing ones. The command-line workflow produces the saved scientific artifacts. `ECE1512_ProjectA_Executed.ipynb` is a review copy that replays the archived checkpoint and recomputes its held-out metrics and overlays, with the full training branch retained. Its execution metadata states this mode explicitly; the six full 30-epoch training runs are recorded separately in results/.

Attention restores the C16 baseline checkpoint on complete bags for test_068(tumor) and test_067(normal). Coordinates are level-0 top-left corners. A512-pixel square footprint is inherited and assumed because extraction metadata are absent. Bright colors mean within-slide attention percentile, not tumor probability. A normal bag can have a hotspot. Two examples do not quantify lesion detection or establish clinical explanations.
