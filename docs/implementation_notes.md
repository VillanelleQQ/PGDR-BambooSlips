# Implementation notes

This reorganization was checked against the locally supplied main manuscript,
*Physics-Guided Multistep Deformation Reversal for Ancient Bamboo Slip
Restoration*, and its supplementary material. The supplied PDFs remain in their
original location and are not required to execute the code.

## Paper-to-code map

| Paper component | Implementation |
| --- | --- |
| Fiber elongation and viscoelastic response (§3, §4.2) | `physics/creep.py`: `CreepDeformationEngine` |
| Stress proxy, local transfer, moisture coupling (§4.2) | `_compute_displacement_gradient`, `_apply_stress_transfer`, `get_incremental_displacement` |
| Progressive forward deformation | `physics/forward.py`: `CreepForwardProcess.forward_step_by_step` |
| ControlPointUNet (§4.3) | `models/control_point_unet.py`: `ControlPointUNet` |
| Single-step displacement MSE (§4.3) | `training/trainer.py`: `CreepDiffusionTrainer` |
| Iterative restoration | `restoration/sampler.py`: `CreepDiffusionSampler` |
| Input preprocessing and dataset split (§5.1) | `data/bamboo.py`, `scripts/train.py` |

## What changed during reorganization

- Split image loading, the physical engine, the forward process, the network,
  training, restoration, and shared utilities into separate modules. Network
  code lives only in `models/`; the duplicate `model/` compatibility directory
  has been removed.
- Replaced unresolved `ddpm.*`/top-level `forward` imports with real module paths.
- Connected model arguments to construction, including control-grid dimensions
  and timestep embeddings. New runs default to `time_emb_dim=256`; use
  `--time_emb_dim 0` to disable conditioning. The original training script did
  not pass this argument, so its effective model had no timestep embedding.
- Set the default learning rate to the manuscript's `1e-4` (the original CLI
  default was `2e-4`). Kept the existing AdamW and cosine schedule.
- Added real log/CSV output, a last checkpoint, serialized configuration, strict
  weight loading, and explicit input/restore failures. Removed unused CLI flags
  rather than suggesting they affect training.
- Allow small datasets without dropping every batch; losses are weighted by the
  number of images. Fixed history sampling for `T < 10`.
- Image export uses Pillow directly, avoiding an otherwise unnecessary torchvision/video dependency.
- Both `python -m scripts.train` and `python scripts/train.py` work from the
  repository root; the sample entry point supports the same invocation styles.

The network's parameter names and physical formulas were retained. New default
training configuration is different from the old script, and scientific results
must be validated again for newly trained models.

## Differences that still require experimental validation

This is a structural cleanup, not a claim that all manuscript details have been
reproduced. The following released behaviors are intentionally recorded:

1. **Forward interpolation:** `_apply_deformation_pytorch` uses inverse-distance
   weights over normalized control-point coordinates. Restoration uses bilinear
   interpolation over the rectangular control grid. The manuscript describes
   bilinear interpolation for the forward field as well.
2. **State evolution:** the supplied forward process warps the previous image
   at each step. Supplementary Eq. (30) instead describes applying cumulative
   displacement to the pristine image. Repeated warping can introduce blur.
3. **Sampling signs:** forward sampling uses `identity + displacement`; training
   uses the negative forward increment as its target; restoration samples with
   `identity - predicted_displacement`. For a constant displacement this would
   move sampling coordinates in the same direction as the forward warp, so
   the convention needs a dedicated numerical and visual evaluation. Negating
   a spatially varying field is also only an approximate inverse warp.
4. **Boundary behavior:** forward sampling uses `border` padding; restoration
   uses `reflection` padding.
5. **Time response:** code uses `1 - exp(-t/tau)` and noise amplitude proportional
   to that response. The manuscript's discrete expression uses `t/T`. The
   original beta schedule buffers remain for API continuity but do not drive
   a Gaussian image-noising process.
6. **Grid and energy:** the implementation creates a rectangular grid spanning
   the image, not a contour-adaptive grid. Energy helper functions are present,
   but no explicit energy minimizer or energy penalty is used by the trainer.
7. **Evaluation:** training reports synthetic displacement MSE. Complete Straightness, LPIPS, TRQ,
   DCI, and baseline evaluation scripts were not in the supplied source and
   have not been invented as part of this reorganization.
8. **Randomness:** small material-heterogeneity noise makes synthetic inputs
   stochastic, including held-out synthetic loss. A seed initializes the run,
   but this does not establish a fixed evaluation dataset or cross-device
   deterministic results.

## Checkpoint format and legacy weights

New checkpoints use tensors and primitive configuration values, allowing
`torch.load(..., weights_only=True)`. They contain:

- `model_state_dict`, `model_config`, `physics_params`, and `T`;
- `epoch` and `test_loss`.

They are inference checkpoints, not full optimizer/scheduler resume snapshots.
Loading is strict: missing or mismatched weights fail instead of being silently
replaced with random parameters.

Old checkpoints lack reliable model configuration and may contain an
`argparse.Namespace`. The new sample command does not guess the architecture.
To migrate a trusted legacy checkpoint, load it separately using its original
trusted environment, reconstruct the **actual training configuration**, and
save the state dict with primitive `model_config`, `physics_params`, and `T`
fields as above. Do not simply copy the legacy CLI namespace: several original
CLI flags were ignored. In particular, the original training constructor used
`time_emb_dim=None`, `dropout=0.1`, and automatically selected attention
resolutions (for a 320×32 image and four levels, `[1, 2, 3]`). Verify dimensions
against the state dict before migration.

## Data and manuscript metadata

The README's dataset sizes and author email addresses come from the supplied
manuscript. Image access is by email due to copyright restrictions, as requested
by the repository owner. No dataset download or pretrained-weight availability
is implied. The README citation uses the CVPR 2026 bibliographic entry supplied
by the repository owner. Dataset access inquiries are directed to the
corresponding author, Yongchao Xu (`yongchao.xu@whu.edu.cn`).
