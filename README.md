# UAV Multi-Modal OBB Detection

RGB-T UAV traffic-scene oriented object detection and staged tracking project.

This repository now uses a simple active-config layout:

- Daily detector training entry: `configs/main/full_project.yaml`
- Detector ablations: `configs/main/baseline.yaml`, `configs/main/fusion_main.yaml`, `configs/main/assigner_main.yaml`, `configs/main/temporal_main.yaml`
- Tracking entries: `configs/main/tracking_base.yaml`, `configs/main/tracking_assoc.yaml`, `configs/main/tracking_temporal.yaml`, `configs/main/tracking_modality.yaml`, `configs/main/tracking_jointlite.yaml`, `configs/main/tracking_final.yaml`, `configs/main/tracking_eval.yaml`

Older `configs/exp_*.yaml` paths are still accepted through config redirects, but they are no longer the recommended daily entry points.

## Recommended Environment

- Python `3.10` recommended
- NVIDIA GPU machine recommended for training
- `requirements.txt` now pins compatible version ranges for the tested runtime stack

Important note about PyTorch:

- `requirements.txt` includes `torch` and `torchvision` for convenience, so on many machines `pip install -r requirements.txt` is enough.
- If the target machine needs a specific CUDA build of PyTorch, install the matching `torch` / `torchvision` wheel first, then run the remaining project install steps.

## Quick Install

Windows PowerShell:

```powershell
git clone https://github.com/timt50084-bot/UAV_MultiModal_Det_demo2.git
cd UAV_MultiModal_Det_demo2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

If you prefer Conda:

```powershell
conda create -n uav-mm python=3.10 -y
conda activate uav-mm
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

## Sanity Check

Run this once after installation:

```powershell
python -c "import torch, cv2, numpy, shapely, omegaconf; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available())"
```

If that succeeds, the runtime dependencies are in place.

## Dataset Preparation

The project expects the processed DroneVehicle layout used by `DroneDualDataset`.

Default processed path in the active config:

- `configs/main/full_project.yaml`
- `dataset.root_dir: 'D:/DataSet/DroneVehicle_Processed'`

If the dataset lives elsewhere on the target machine, either:

1. Edit `dataset.root_dir` in `configs/main/full_project.yaml`, or
2. Override it at launch time with CLI:

```powershell
python tools/prepare_dronevehicle_dataset.py --raw-root E:/DroneVehicle --output-root E:/DroneVehicle_Processed --splits train val
python tools/train.py --config configs/main/full_project.yaml --device 0 dataset.root_dir=E:/DroneVehicle_Processed
```

Standard preparation command:

```powershell
python tools/prepare_dronevehicle_dataset.py --raw-root D:/DataSet/DroneVehicle --output-root D:/DataSet/DroneVehicle_Processed --splits train val
```

## Final Detector Training

The final detector mainline is:

- `configs/main/full_project.yaml`

Typical full training command:

```powershell
python tools/train.py --config configs/main/full_project.yaml --device 0
```

Quick smoke run:

```powershell
python tools/train.py --config configs/main/full_project.yaml --device 0 train.epochs=1
```

The current active full-project config is intended to hold the day-to-day training knobs in one place:

- `dataloader.batch_size`
- `dataloader.num_workers`
- `dataset.imgsz`
- `train.epochs`
- `train.patience`
- `train.lr`
- `train.weight_decay`
- `train.accumulate`

At startup the training entry prints the effective config summary and also saves:

- `outputs/experiments/<run_name>/resolved_config.yaml`

That file is the final merged config actually used for the run.

## Detector Ablation Runs

These configs inherit the common training schedule and dataloader settings from `configs/main/full_project.yaml`.

Use these entries:

- `configs/main/baseline.yaml`
- `configs/main/fusion_main.yaml`
- `configs/main/assigner_main.yaml`
- `configs/main/temporal_main.yaml`
- `configs/main/full_project.yaml`

Recommended commands:

```powershell
python tools/train.py --config configs/main/baseline.yaml --device 0
python tools/train.py --config configs/main/fusion_main.yaml --device 0
python tools/train.py --config configs/main/assigner_main.yaml --device 0
python tools/train.py --config configs/main/temporal_main.yaml --device 0
python tools/train.py --config configs/main/full_project.yaml --device 0
```

Recommended workflow:

1. Change shared training hyperparameters only in `configs/main/full_project.yaml`
2. Switch the `--config` path when you want to run a detector ablation
3. Keep the ablation file focused on feature on/off choices

## Validation

```powershell
python tools/val.py --config configs/main/full_project.yaml --weights outputs/experiments/full_project/weights/best.pt --device 0
```

## Tracking Entry Points

Active tracking configs are also under `configs/main/`:

- `configs/main/tracking_base.yaml`
- `configs/main/tracking_assoc.yaml`
- `configs/main/tracking_temporal.yaml`
- `configs/main/tracking_modality.yaml`
- `configs/main/tracking_jointlite.yaml`
- `configs/main/tracking_final.yaml`
- `configs/main/tracking_eval.yaml`

Example tracking inference:

```powershell
python tools/infer.py --config configs/main/tracking_final.yaml --weights outputs/experiments/full_project/weights/best.pt --source_rgb path/to/rgb_frames --source_ir path/to/ir_frames --save_dir outputs/tracking_final
```

## Minimal Pre-Run Checklist For Another Machine

Before you leave for the other computer, this is the shortest reliable checklist:

1. Copy the repository
2. Install the environment with `requirements.txt`
3. Run the import sanity check
4. Prepare or point to the processed dataset
5. Open `configs/main/full_project.yaml`
6. Confirm:
   - `dataset.root_dir`
   - `dataloader.batch_size`
   - `dataset.imgsz`
   - `train.epochs`
7. Run a 1-epoch smoke command first
8. Then launch full training

If you want the least confusion on the target machine, ignore old `configs/exp_*.yaml` paths and use only `configs/main/*.yaml`.
