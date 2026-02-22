import argparse
import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from ddpm.Unet import ControlPointUNet
from ddpm.diffusion import CreepDiffusionTrainer, create_trainer
from ddpm.forward import CreepDeformationEngine


class BambooSlipsDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_size=(320, 32)):
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        self.img_paths = []
        for ext in valid_extensions:
            self.img_paths.extend(
                [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                 if f.lower().endswith(ext)]
            )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.image_size[1], self.image_size[0]), Image.LANCZOS)
            if self.transform:
                img_tensor = self.transform(img)
            else:
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        except Exception as e:
            img_tensor = torch.zeros(3, self.image_size[0], self.image_size[1])
        return img_tensor


class TrainingLogger:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.log_dir / "checkpoints"
        self.visualizations_dir = self.log_dir / "visualizations"
        for dir_path in [self.checkpoints_dir, self.visualizations_dir]:
            dir_path.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "training.log"
        self.metrics_file = self.log_dir / "metrics.csv"
        self.steps = []
        self.losses = []
        with open(self.log_file, 'w', encoding='utf-8') as f:
            pass
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            pass

    def log(self, message, step=None):
        pass

    def log_metrics(self, step, loss):
        self.steps.append(step)
        self.losses.append(loss)

    def plot_training_curves(self):
        pass


def create_trainer(model, T=100, image_size=(320, 32), **physics_params):
    return CreepDiffusionTrainer(
        model=model,
        beta_1=1e-4,
        beta_T=0.02,
        T=T,
        image_size=image_size,
        control_grid_size=(32, 4),
        **physics_params
    )


def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"creep_diffusion_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = TrainingLogger(output_dir)

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = BambooSlipsDataset(
        root_dir=str(Path(args.dataset_root) / "train"),
        transform=transform,
        image_size=args.image_size
    )
    test_dataset = BambooSlipsDataset(
        root_dir=str(Path(args.dataset_root) / "test"),
        transform=transform,
        image_size=args.image_size
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                 pin_memory=True)

    model = ControlPointUNet(
        img_channels=args.img_channels,
        base_channels=args.base_channels,
        image_size=args.image_size,
        control_grid_size=(args.control_ny, args.control_nx)
    )

    physics_params = {
        'fiber_elongation_factor': args.fiber_elongation_factor,
        'force_coupling_strength': args.force_coupling_strength,
        'moisture_diffusion_coeff': args.moisture_diffusion_coeff,
        'inertia_factor': args.inertia_factor,
        'elastic_modulus': args.elastic_modulus,
        'viscosity_coeff': args.viscosity_coeff,
        'relaxation_time': args.relaxation_time,
        'orientation_response_coeff': args.orientation_response_coeff,
        'orientation_damping': args.orientation_damping,
    }

    trainer = create_trainer(model, T=args.T, image_size=args.image_size, **physics_params)

    model.to(device)
    trainer.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.1)

    global_step = 0
    best_loss = float('inf')

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_dataloader)

        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            loss = trainer(batch).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % args.log_interval == 0:
                logger.log_metrics(global_step, loss.item())

        scheduler.step()
        avg_epoch_loss = epoch_loss / len(train_dataloader)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for test_batch in test_dataloader:
                test_batch = test_batch.to(device)
                test_loss += trainer(test_batch).mean().item()
        avg_test_loss = test_loss / len(test_dataloader)

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_model_path = logger.checkpoints_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'test_loss': best_loss,
                'args': args,
            }, best_model_path)

    logger.plot_training_curves()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--image_size', type=int, nargs=2, default=[320, 32])
    parser.add_argument('--img_channels', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--channel_mults', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--time_emb_dim', type=int, default=256)
    parser.add_argument('--time_emb_scale', type=float, default=1.0)
    parser.add_argument('--num_classes', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--attention_resolutions', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--norm', type=str, default='gn')
    parser.add_argument('--num_groups', type=int, default=32)
    parser.add_argument('--max_displacement', type=float, default=50.0)
    parser.add_argument('--control_nx', type=int, default=4)
    parser.add_argument('--control_ny', type=int, default=32)
    parser.add_argument('--beta_1', type=float, default=1e-4)
    parser.add_argument('--beta_T', type=float, default=0.02)
    parser.add_argument('--T', type=int, default=150)
    parser.add_argument('--fiber_elongation_factor', type=float, default=0.15)
    parser.add_argument('--force_coupling_strength', type=float, default=0.3)
    parser.add_argument('--moisture_diffusion_coeff', type=float, default=0.15)
    parser.add_argument('--em_modulus', type=float, default=0.6)
    parser.add_argument('--viscosity', type=float, default=12.0)
    parser.add_argument('--time_step', type=float, default=1.0)
    parser.add_argument('--max_physics_iterations', type=int, default=20)
    parser.add_argument('--convergence_threshold', type=float, default=0.01)
    parser.add_argument('--boundary_factor', type=float, default=0.6)
    parser.add_argument('--inertia_factor', type=float, default=0.8)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--sample_interval', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--plot_interval', type=int, default=100)
    parser.add_argument('--elastic_modulus', type=float, default=0.6)
    parser.add_argument('--viscosity_coeff', type=float, default=12.0)
    parser.add_argument('--relaxation_time', type=float, default=None)
    parser.add_argument('--orientation_response_coeff', type=float, default=0.1)
    parser.add_argument('--orientation_damping', type=float, default=0.8)

    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    train_model(args)


if __name__ == '__main__':
    main()