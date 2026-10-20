# PPFN-small-direct — Real-Time Thermal Image Restoration

A lightweight (4.26M param) restoration model for 16-bit raw thermal camera
frames, distilled down from [PPFN](https://arxiv.org/abs/2111.09881)
(Restormer-based, NeurIPS 2025) to hit real-time framerates while removing
sensor noise, motion blur, and illumination degradation in a single forward
pass.

- **66.9 FPS** with fp16 + `torch.compile` on a single 24GB GPU
- Trained from random initialization on a public dataset only — no
  project-specific captured data required
- Training and inference code are fully decoupled; they share only the
  model architecture and a checkpoint file

## Structure

```
.
├── model/
│   └── model.py          Restormer_PPFN architecture
├── train/
│   └── train.py           Training script
├── infer/
│   ├── infer.py            RealtimeRestorer inference class / CLI
│   └── data_io.py           raw16 normalization + lens undistortion
├── prompts/
│   ├── type7.npy            Prompt embeddings required by the model
│   └── multi2.npy
├── checkpoints/
│   └── ppfn_small_direct_final.pth
├── Dockerfile
└── requirements.txt
```

`train/` and `infer/` do not import each other — they are connected only
through a `.pth` checkpoint file, so the inference code can be deployed
independently of the training pipeline.

## Setup

```bash
pip install -r requirements.txt
```

or build the Docker image:

```bash
docker build -t ppfn-small-direct .
```

## Dataset

Training uses the public **HM-TIR** clean image corpus (NeurIPS 2025, PPFN
paper), available on
[Google Drive](https://drive.google.com/file/d/17kq2NQ-8bFeESlanffcOSlFLwGb0Y8Rq/view).
Extract it so that `HM-TIR/imgs/*.png` (1503 images) exists under a chosen
parent directory, which is passed as `--clean-data-root`.

```
<clean-data-root>/
└── HM-TIR/
    └── imgs/
        ├── 00000.png
        └── ...
```

## Training

```bash
cd train
python train.py --clean-data-root /path/to/HM-TIR-parent --out-dir ../checkpoints
```

Default arguments reproduce `ppfn_small_direct_v1` exactly: random
initialization, `dim=24`, `num_blocks=[2,3,3,4]`, `num_refinement_blocks=2`,
`lr=2e-4`, 30000 steps, batch size 4, patch size 256 (~2.5 hours on an RTX
4090). Synthetic degradation (noise + motion blur + illumination, each with
an independent "sometimes none" branch) is generated on the fly — no
paired/real-world capture data is needed.

## Inference

```bash
cd infer
python infer.py raw_frame.png restored.png --ckpt ../checkpoints/ppfn_small_direct_final.pth

# with optional lens-distortion correction and 2-point NUC calibration:
python infer.py raw_frame.png restored.png \
    --ckpt ../checkpoints/ppfn_small_direct_final.pth \
    --camchain camchain.yaml --nuc-h5 two_point_nuc.h5
```

Programmatic use:

```python
from infer import RealtimeRestorer

restorer = RealtimeRestorer('../checkpoints/ppfn_small_direct_final.pth')
restored01 = restorer.restore_frame(raw16)  # raw16: uint16 numpy array -> float32 [0, 1]
```

## Pipeline

```
raw16 -> [optional: 2-point NUC] -> PPFN-small-direct (single pass) -> [optional: lens undistort]
```

Undistortion is applied to the model's output, not its input: the model is
trained and validated entirely on the distorted pixel grid, so undistorting
the input first would push it off-distribution. Geometric correction is a
closed-form transform and is kept as a separate, deterministic final step.

## Acknowledgements

The model architecture (`model/model.py`) is copied from the [HM-TIR / PPFN](https://arxiv.org/abs/2111.09881)
repository (NeurIPS 2025), itself based on
[Restormer](https://arxiv.org/abs/2111.09881).
