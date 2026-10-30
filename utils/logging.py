"""Persist training progress without optional plotting dependencies."""
import csv
import logging
from pathlib import Path


def create_logger(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(output_dir.resolve()))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        for handler in (logging.StreamHandler(), logging.FileHandler(output_dir / 'training.log')):
            handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
            logger.addHandler(handler)
    return logger


def append_metrics(path, epoch, train_loss, test_loss, lr):
    path = Path(path)
    new_file = not path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.writer(stream)
        if new_file:
            writer.writerow(['epoch', 'train_loss', 'test_loss', 'learning_rate'])
        writer.writerow([epoch, train_loss, test_loss, lr])
