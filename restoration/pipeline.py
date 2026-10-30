"""Load a checkpoint, restore an RGB image, and save reviewable outputs."""
from pathlib import Path

import torch
from utils.images import save_image

from data.bamboo import load_image
from restoration.sampler import CreepDiffusionSampler
from utils.checkpoint import load_model


class BambooRestorer:
    def __init__(self, model_path, device='cuda', T=None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model, checkpoint = load_model(model_path, self.device)
        self.T = checkpoint['T'] if T is None else T
        self.image_size = tuple(checkpoint['model_config']['image_size'])
        self.sampler = CreepDiffusionSampler(self.model, T=self.T).to(self.device)

    def load_image(self, path, target_size=None):
        if target_size is not None and tuple(target_size) != self.image_size:
            raise ValueError(f'Checkpoint requires image_size={self.image_size}')
        return load_image(path, self.image_size).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def restore_single_image(self, image, show_progress=True, save_history=False):
        self.model.eval()
        return self.sampler(image, show_progress=show_progress, save_history=save_history)

    @staticmethod
    def save_results(deformed, restored, history, image_name, output_dir):
        output_dir = Path(output_dir)
        for folder in ('restored', 'comparisons'):
            (output_dir / folder).mkdir(parents=True, exist_ok=True)
        deformed = deformed.squeeze(0).clamp(0, 1)
        restored = restored.squeeze(0).clamp(0, 1)
        save_image(restored, output_dir / 'restored' / f'{image_name}_restored.png')
        # CHW: concatenate along width for a side-by-side comparison.
        save_image(torch.cat([deformed, restored], dim=2),
                   output_dir / 'comparisons' / f'{image_name}_comparison.png')
        if history:
            history_dir = output_dir / 'history' / f'{image_name}_frames'
            history_dir.mkdir(parents=True, exist_ok=True)
            for index, frame in enumerate(history):
                save_image(frame.clamp(0, 1), history_dir / f'frame_{index:03d}.png')
