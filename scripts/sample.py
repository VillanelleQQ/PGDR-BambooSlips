import argparse
import os
import torch
import torchvision
import numpy as np
from PIL import Image
from pathlib import Path
import datetime
import sys
import traceback

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ddpm.Unet import ControlPointUNet
from ddpm.diffusion import CreepDiffusionSampler


class BambooRestorer:
    def __init__(self, model_path, device='cuda', T=150):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.T = T
        self.model = self._load_model(model_path)
        self.sampler = CreepDiffusionSampler(self.model, T=self.T)
        self.sampler.to(self.device)

    def _load_model(self, model_path):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
        except Exception as e:
            raise

        model = ControlPointUNet(
            img_channels=3,
            base_channels=64,
            control_grid_size=(32, 4),
            image_size=(320, 32),
            channel_mults=(1, 2, 4, 8),
            num_res_blocks=2,
            time_emb_dim=None,
            time_emb_scale=1.0,
            num_classes=None,
            activation=torch.nn.functional.relu,
            dropout=0.2,
            attention_resolutions=(1, 2),
            norm="gn",
            num_groups=32,
            max_displacement=50.0,
        )

        if 'model_state_dict' in checkpoint:
            model_dict = model.state_dict()
            pretrained_dict = checkpoint['model_state_dict']
            filtered_dict = {k: v for k, v in pretrained_dict.items()
                             if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()
        return model

    def load_image(self, image_path, target_size=(320, 32)):
        try:
            pil_image = Image.open(image_path).convert('RGB')
            original_size = pil_image.size
            pil_image = pil_image.resize((target_size[1], target_size[0]), Image.LANCZOS)
            img_array = np.array(pil_image, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
            return img_tensor.to(self.device)
        except Exception as e:
            return None

    def restore_single_image(self, deformed_image_tensor, show_progress=True):
        self.model.eval()
        with torch.no_grad():
            restored_tensor, history = self.sampler(deformed_image_tensor, show_progress=show_progress)
            return restored_tensor, history

    @staticmethod
    def _save_restoration_results(deformed_tensor, restored_tensor, history, image_name, output_dir):
        restored_dir = output_dir / "restored"
        comparisons_dir = output_dir / "comparisons"
        history_dir = output_dir / "history"
        for dir_path in [restored_dir, comparisons_dir, history_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)

        deformed_tensor = torch.clamp(deformed_tensor, 0, 1)
        restored_tensor = torch.clamp(restored_tensor, 0, 1)

        torchvision.utils.save_image(
            restored_tensor,
            restored_dir / f"{image_name}_restored.png",
            normalize=False
        )
        comparison_image = torch.cat([deformed_tensor, restored_tensor], dim=2)
        torchvision.utils.save_image(
            comparison_image,
            comparisons_dir / f"{image_name}_comparison.png",
            normalize=False
        )

        if history:
            frame_history_dir = history_dir / f"{image_name}_frames"
            frame_history_dir.mkdir(exist_ok=True, parents=True)
            for i, frame_tensor in enumerate(history):
                frame_filename = frame_history_dir / f"{image_name}_history_{i:03d}.png"
                if frame_tensor.dim() == 4:
                    frame_tensor = frame_tensor.squeeze(0)
                torchvision.utils.save_image(
                    torch.clamp(frame_tensor, 0, 1),
                    frame_filename,
                    normalize=False
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,)
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--T', type=int, default=150)
    parser.add_argument('--image_size', type=int, nargs=2, default=[320, 32])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--save_history', action='store_true')
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        return

    model_path = Path(args.model_path)
    if not model_path.exists():
        return

    try:
        restorer = BambooRestorer(
            model_path=args.model_path,
            device=args.device,
            T=args.T
        )
    except Exception as e:
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"restoration_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        image_paths = [input_path]
    else:
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        image_paths = [
            p for p in input_path.rglob('*')
            if p.suffix.lower() in valid_extensions
        ]

    if not image_paths:
        return

    success_count = 0
    total_count = len(image_paths)

    for i, image_path in enumerate(image_paths, 1):
        image_name = image_path.stem
        deformed_tensor = restorer.load_image(image_path, tuple(args.image_size))
        if deformed_tensor is None:
            continue
        try:
            restored_tensor, history = restorer.restore_single_image(
                deformed_tensor,
                show_progress=True
            )
            BambooRestorer._save_restoration_results(
                deformed_tensor.squeeze(0),
                restored_tensor.squeeze(0),
                history if args.save_history else None,
                image_name,
                output_dir
            )
            success_count += 1
        except Exception as e:
            pass

if __name__ == "__main__":
    main()