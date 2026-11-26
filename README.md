# Fish species identification and health screening from DBBD-compressed images

Code for the paper

> Manohar K and Santhakumar Mohan, "Joint Fish Species Identification and Health Screening Directly from DBBD-Compressed Images," submitted to SURAS 2027 – Symposium on Underwater Robotics and Autonomous Systems.

A single MobileNetV2 with two heads (12 freshwater species; healthy / bacterial gill disease / bacterial
red disease / EUS) is trained directly on images compressed by a deterministic binary block-diagonal
(DBBD) operator, with no reconstruction. The code runs the compression study at block sizes b = 1, 2, 4
(100%, 25% and 6.25% of the measurements), five-fold fish-level cross-validation, an input-size control,
and a leave-one-disease-out test of abnormality scores.

## Data

Download the two public datasets and note their folders:

| Dataset | Content | Source |
|---|---|---|
| BD-Freshwater-Fish | 12 species, 4,389 images | Das et al., *Data in Brief* 57, 111132 (2024) |
| MatsyaDx-BD (version 2) | healthy, BGD, BRD, EUS; 2,137 images of 86 farm fish | Salam et al., Mendeley Data, doi:10.17632/sxkynv9t7n.2 |

Images are not included in this repository.

## Setup

```
conda create -n dbbd python=3.11 -y
conda activate dbbd
pip install -r requirements.txt
```

Training runs on the CPU (about 2–4 h for the main experiments on a desktop).

## Running

Open `run_all.py`, set `BDFF` and `MATSYA` to the dataset folders, choose `MODE`, and run it
(`python run_all.py`, or F5 in Spyder). Every step is skipped if its output already exists, so a run can be
stopped and restarted.

| MODE | What it does | Time (desktop CPU) |
|---|---|---|
| `check` | end-to-end test on synthetic data | 3–5 min |
| `full` | data preparation, models at b = 1, 2, 4, TFLite export, leave-one-disease-out | 2–4 h |
| `reeval` | re-scores trained models only | minutes |
| `extra` | 5-fold cross-validation by fish; enlarged-input control; extra abnormality scores | 2–2.5 h |
| `upcv` | enlarged-input control (b = 4) in the same 5 folds | 1 h |

Results go to `runs/` (metrics, test scores, models) and `results/` (tables, bootstrap intervals, figures).
On Linux, `bash run_all.sh` runs the `full` steps and `bash selftest.sh` the synthetic check.

## Files

| File | Purpose |
|---|---|
| `dbbd.py` | DBBD operator, Y = A X Aᵀ on 32×32 patches |
| `prepare_data.py` | builds the labelled dataset; groups near-duplicate photos by fish (dHash) and splits by group |
| `make_folds.py` | five label-stratified, fish-level folds |
| `train.py` | two-head MobileNetV2 with masked, class-weighted loss |
| `evaluate.py` | accuracy, macro-F1, sensitivity, specificity, AUROC and abnormality scores |
| `export_tflite.py` | float32 and int8 TensorFlow Lite export |
| `model_cost.py` | parameters and MFLOPs |
| `make_paper_numbers.py` | fish-level bootstrap intervals, tables and figures |
| `make_synthetic_data.py` | synthetic images for the self-test |
| `run_all.py`, `run_all.sh`, `selftest.sh` | run everything |
