"""Restore one image or a directory using a PGDR checkpoint."""
import argparse
import datetime
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', type=Path, required=True)
    parser.add_argument('--input_path', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, default=Path('outputs/restoration'))
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--T', type=int, default=None, help='Defaults to the checkpoint timestep count')
    parser.add_argument('--save_history', action='store_true')
    args = parser.parse_args()
    if args.T is not None and args.T < 1:
        parser.error('T must be positive')
    if not args.model_path.is_file():
        parser.error(f'Checkpoint does not exist: {args.model_path}')
    if not args.input_path.exists():
        parser.error(f'Input does not exist: {args.input_path}')
    return args


def main():
    args = parse_args()
    from data.bamboo import image_paths
    from restoration.pipeline import BambooRestorer

    paths = image_paths(args.input_path)
    restorer = BambooRestorer(args.model_path, args.device, args.T)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_dir = args.output_dir / f'restoration_{stamp}'
    success_count = 0
    for path in paths:
        # Retain relative folders and extensions so repeated stems do not overwrite.
        relative = Path(path.name) if args.input_path.is_file() else path.relative_to(args.input_path)
        try:
            image = restorer.load_image(path)
            restored, history = restorer.restore_single_image(image, save_history=args.save_history)
            restorer.save_results(image, restored, history, relative.name, output_dir / relative.parent)
            success_count += 1
        except Exception as error:
            print(f'Failed to restore {path}: {error}', file=sys.stderr)
    print(f'Restored {success_count}/{len(paths)} images. Outputs: {output_dir}')
    if success_count != len(paths):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
