"""Load model weights strictly instead of silently skipping parameters."""
import torch
from models.control_point_unet import ControlPointUNet
from utils.config import validate_model_config


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or 'model_config' not in checkpoint:
        raise ValueError(
            'Checkpoint has no model_config. Use a checkpoint saved by the reorganized '
            'training entry point, or migrate the legacy checkpoint as described '
            'in docs/implementation_notes.md.'
        )
    config = validate_model_config(dict(checkpoint['model_config']))
    model = ControlPointUNet(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return model, checkpoint
