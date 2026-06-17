# FedMoSA: Federated Medical Image Segmentation under Modality Heterogeneity via Specialized Adapters

This repository provides the official implementation of **FedMoSA**, a federated learning framework designed to address **modality heterogeneity** across distributed clients for medical image segmentation.

The framework supports segmentation experiments on medical imaging datasets (liver, brain) and enables fair comparison with several federated learning baselines.

---

## Dataset Preparation

The `dataset/` directory contains scripts to generate datasets for federated learning experiments. Each script prepares client-wise data splits suitable for modality-heterogeneous federated settings.

### Available Dataset Scripts

- `generate_liver_seg.py`
- `generate_brain_seg.py`

### Generating a Dataset

From the `dataset/` directory, run:

```bash
python3 <SCRIPT_NAME>
```

Example:

```bash
python3 generate_liver_seg.py
```

Within each dataset script, the following parameters can be modified:

- Number of clients
- Number of runs
- Dataset split configuration
- Output directory paths

All parameters are defined directly inside the corresponding script.

---

## Running Experiments

All federated learning experiments are executed from the `system/` directory using the provided shell script.

### Main Components

- `main.py` — Core federated learning training entry point
- `FedSAM.sh` — Script to launch experiments sequentially
- `system/FedSAM/` — FedSAM method implementations
- `system/flcore/` — Core federated learning utilities
- `system/utils/` — Helper utilities

### Running Experiments

```bash
cd system
bash FedSAM.sh
```

Hyperparameters for each method and dataset can be adjusted directly inside `FedSAM.sh`.

---

## Example Commands

### FedMoSA (proposed)

```bash
nohup python3 main.py \
  -algo FedMoSA \
  -lr 0.0001 \
  -m mosa \
  -mn mosa \
  -nc 5 \
  -data liver_seg \
  -pr 0 \
  -pv 0 \
  -t 1 \
  -go experiment \
  -gpu 0 \
  > FedSAM_liver_5_mosa_1.log 2>&1 &
```

### FednnUNET with Adapter Fine-tuning (AFA)

```bash
nohup python3 main.py \
  -algo FednnUNET \
  -lr 0.01 \
  -m nnunet \
  -mn nnunet \
  -nc 5 \
  -data liver_seg \
  -afa 1 \
  -pr 0 \
  -pv 0 \
  -t 1 \
  -go experiment-afa \
  -gpu 0 \
  > FednnUNET_liver_5_nnunet_1_AFA.log 2>&1 &
```

---

## Key Arguments

| Argument | Description |
|----------|-------------|
| `-algo`  | Federated algorithm to use (e.g., `FedMoSA`, `FedMSA`, `FednnUNET`) |
| `-lr`    | Learning rate |
| `-m`     | Model architecture |
| `-mn`    | Model name |
| `-nc`    | Number of clients |
| `-data`  | Dataset to use (`liver_seg`, `brain_seg`) |

---

## Supported Methods

- **FedMoSA** (proposed) — Federated learning with specialized and shared adapters
- **FedMSA** — Federated learning with medical SAM adapter
- **FednnUNET FFE** — FednnU-Net with federated fingerprint extraction
- **FednnUNET AFA** — FednnU-NET with asymmetric federated averaging

---

## Results

Experiment outputs and logs are saved to the `results/` directory. Training logs are written to `.log` files in the `system/` directory for later inspection.

---

## Notes

- Experiments are executed asynchronously using `nohup`.
- Training logs are saved to `.log` files (e.g., `FedSAM_liver_5_mosa_1.log`).
- GPU selection is controlled via the `-gpu` argument.
