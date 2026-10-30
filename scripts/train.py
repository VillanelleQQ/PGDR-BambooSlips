"""Train PGDR with online physics-generated deformation pairs."""
import argparse
import datetime
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_root', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, default=Path('outputs'))
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--image_size', type=int, nargs=2, default=[320, 32], metavar=('HEIGHT', 'WIDTH'))
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--channel_mults', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--time_emb_dim', type=int, default=256, help='Use 0 to disable time conditioning')
    parser.add_argument('--time_emb_scale', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--attention_resolutions', type=int, nargs='*', default=[1, 2])
    parser.add_argument('--num_groups', type=int, default=32)
    parser.add_argument('--max_displacement', type=float, default=50.0)
    parser.add_argument('--control_nx', type=int, default=4)
    parser.add_argument('--control_ny', type=int, default=32)
    parser.add_argument('--T', type=int, default=150)
    parser.add_argument('--fiber_elongation_factor', type=float, default=0.15)
    parser.add_argument('--force_coupling_strength', type=float, default=0.3)
    parser.add_argument('--moisture_diffusion_coeff', type=float, default=0.15)
    parser.add_argument('--elastic_modulus', type=float, default=0.6)
    parser.add_argument('--viscosity_coeff', type=float, default=12.0)
    parser.add_argument('--relaxation_time', type=float, default=None)
    parser.add_argument('--orientation_response_coeff', type=float, default=0.1)
    parser.add_argument('--orientation_damping', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.T, args.num_epochs, args.batch_size) < 1 or args.num_workers < 0:
        parser.error('T, num_epochs and batch_size must be positive; num_workers must be nonnegative')
    if min(args.elastic_modulus, args.viscosity_coeff, args.orientation_damping) <= 0:
        parser.error('Elastic modulus, viscosity and orientation damping must be positive')
    if args.relaxation_time is not None and args.relaxation_time <= 0:
        parser.error('relaxation_time must be positive')
    return args


def train_model(args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from data.bamboo import BambooSlipsDataset
    from models.control_point_unet import ControlPointUNet
    from training.trainer import CreepDiffusionTrainer
    from utils.config import MODEL_DEFAULTS, validate_model_config
    from utils.logging import append_metrics, create_logger

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model_config = dict(MODEL_DEFAULTS)
    for key in model_config:
        if hasattr(args, key):
            model_config[key] = getattr(args, key)
    model_config['time_emb_dim'] = args.time_emb_dim or None
    model_config['control_grid_size'] = [args.control_ny, args.control_nx]
    validate_model_config(model_config)
    physics_keys = (
        'fiber_elongation_factor', 'force_coupling_strength', 'moisture_diffusion_coeff',
        'elastic_modulus', 'viscosity_coeff', 'relaxation_time',
        'orientation_response_coeff', 'orientation_damping',
    )
    physics_params = {key: getattr(args, key) for key in physics_keys}
    loaders = {}
    for split in ('train', 'test'):
        dataset = BambooSlipsDataset(args.dataset_root / split, image_size=args.image_size)
        loaders[split] = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=split == 'train',
            num_workers=args.num_workers, pin_memory=device.type == 'cuda', drop_last=False,
        )
    model = ControlPointUNet(**model_config).to(device)
    trainer = CreepDiffusionTrainer(
        model, beta_1=1e-4, beta_T=0.02, T=args.T,
        image_size=args.image_size, control_grid_size=model_config['control_grid_size'],
        **physics_params,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.1
    )
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_dir = args.output_dir / f'pgdr_{stamp}'
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True)
    logger = create_logger(output_dir)
    config = {'model_config': model_config, 'physics_params': physics_params, 'T': args.T,
              'training': {key: str(v) if isinstance(v, Path) else v for key, v in vars(args).items()}}
    (output_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    logger.info('Device: %s; outputs: %s', device, output_dir)
    best_loss = float('inf')
    for epoch in range(1, args.num_epochs + 1):
        losses = {}
        for split, loader in loaders.items():
            model.train(split == 'train')
            total, count = 0.0, 0
            with torch.set_grad_enabled(split == 'train'):
                for batch in tqdm(loader, desc=f'Epoch {epoch}/{args.num_epochs} {split}'):
                    batch = batch.to(device)
                    if split == 'train':
                        optimizer.zero_grad()
                    loss = trainer(batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Non-finite {split} loss at epoch {epoch}')
                    if split == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                        optimizer.step()
                    total += loss.item() * batch.shape[0]
                    count += batch.shape[0]
            losses[split] = total / count
        lr = optimizer.param_groups[0]['lr']
        append_metrics(output_dir / 'metrics.csv', epoch, losses['train'], losses['test'], lr)
        logger.info('Epoch %d train=%.6g test=%.6g lr=%.6g', epoch, losses['train'], losses['test'], lr)
        scheduler.step()
        checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                      'model_config': model_config, 'physics_params': physics_params,
                      'T': args.T, 'test_loss': losses['test']}
        torch.save(checkpoint, checkpoint_dir / 'last_model.pt')
        if losses['test'] < best_loss:
            best_loss = losses['test']
            torch.save(checkpoint, checkpoint_dir / 'best_model.pt')


def main():
    train_model(parse_args())


if __name__ == '__main__':
    main()
