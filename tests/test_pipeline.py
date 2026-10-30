"""Integration checks on generated patterns; no restricted images are used."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from data.bamboo import BambooSlipsDataset
from restoration.pipeline import BambooRestorer
from scripts.train import parse_args, train_model
from utils.checkpoint import load_model


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        # RGB bands provide nonconstant structure for a real backward/warp pass.
        for split, count in [('train', 3), ('test', 1), ('real_deformed', 1)]:
            directory = cls.root / 'data' / split
            directory.mkdir(parents=True)
            for index in range(count):
                image = np.zeros((16, 8, 3), dtype=np.uint8)
                image[:, ::2, :] = 180 + index
                image[::3, :, 1] = 255
                Image.fromarray(image).save(directory / f'slip_{index}.png')
        argv = [
            'train', '--dataset_root', str(cls.root / 'data'),
            '--output_dir', str(cls.root / 'outputs'), '--device', 'cpu',
            '--image_size', '16', '8', '--control_ny', '4', '--control_nx', '2',
            '--base_channels', '8', '--channel_mults', '1', '2',
            '--num_groups', '4', '--num_res_blocks', '1', '--time_emb_dim', '16',
            '--attention_resolutions', '1', '--batch_size', '2', '--num_workers', '0',
            '--num_epochs', '1', '--T', '1',
        ]
        with patch('sys.argv', argv):
            train_model(parse_args())
        cls.run_dir = next((cls.root / 'outputs').glob('pgdr_*'))
        cls.checkpoint = cls.run_dir / 'checkpoints' / 'best_model.pt'

    @classmethod
    def tearDownClass(cls):
        import logging
        logger = logging.getLogger(str(cls.run_dir.resolve()))
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        cls.temp.cleanup()

    def test_training_artifacts_and_partial_batch(self):
        self.assertTrue(self.checkpoint.is_file())
        self.assertTrue((self.run_dir / 'checkpoints' / 'last_model.pt').is_file())
        config = json.loads((self.run_dir / 'config.json').read_text())
        self.assertEqual(config['model_config']['control_grid_size'], [4, 2])
        self.assertEqual(config['model_config']['time_emb_dim'], 16)
        lines = (self.run_dir / 'metrics.csv').read_text().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(np.isfinite(float(lines[1].split(',')[1])))

    def test_checkpoint_restoration_and_history(self):
        restorer = BambooRestorer(self.checkpoint, device='cpu')
        self.assertEqual(restorer.image_size, (16, 8))
        self.assertEqual(restorer.T, 1)
        image = restorer.load_image(self.root / 'data/real_deformed/slip_0.png')
        restored, history = restorer.restore_single_image(image, show_progress=False, save_history=True)
        self.assertEqual(tuple(restored.shape), (1, 3, 16, 8))
        self.assertEqual(len(history), 2)
        self.assertTrue(torch.isfinite(restored).all())
        self.assertTrue(((restored >= 0) & (restored <= 1)).all())
        destination = self.root / 'restored'
        restorer.save_results(image, restored, history, 'slip_0.png', destination)
        with Image.open(destination / 'comparisons/slip_0.png_comparison.png') as comparison:
            self.assertEqual(comparison.size, (16, 16))
        self.assertEqual(len(list((destination / 'history/slip_0.png_frames').glob('*.png'))), 2)
        _, no_history = restorer.restore_single_image(image, show_progress=False)
        self.assertEqual(no_history, [])

    def test_checkpoint_rejects_missing_weights(self):
        checkpoint = torch.load(self.checkpoint, weights_only=True)
        del checkpoint['model_state_dict']['init_conv.weight']
        path = self.root / 'bad_checkpoint.pt'
        torch.save(checkpoint, path)
        with self.assertRaises(RuntimeError):
            load_model(path, 'cpu')

    def test_dataset_errors_are_visible(self):
        with self.assertRaises(FileNotFoundError):
            BambooSlipsDataset(self.root / 'missing')
        directory = self.root / 'corrupt'
        directory.mkdir(exist_ok=True)
        (directory / 'bad.png').write_bytes(b'not an image')
        dataset = BambooSlipsDataset(directory)
        with self.assertRaises(OSError):
            dataset[0]

    def test_directory_cli(self):
        from scripts.sample import main
        source = self.root / 'data/real_deformed'
        nested = source / 'nested'
        nested.mkdir(exist_ok=True)
        # Same stem in another folder must produce a separate output.
        with Image.open(source / 'slip_0.png') as image:
            image.save(nested / 'slip_0.png')
        destination = self.root / 'cli_outputs'
        argv = ['sample', '--model_path', str(self.checkpoint), '--input_path', str(source),
                '--output_dir', str(destination), '--device', 'cpu']
        with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()) as output:
            main()
        self.assertIn('Restored 2/2', output.getvalue())
        self.assertEqual(len(list(destination.rglob('*_restored.png'))), 2)


if __name__ == '__main__':
    unittest.main()
