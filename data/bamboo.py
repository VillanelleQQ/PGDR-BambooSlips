"""Image loading shared by training and restoration."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def image_paths(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Image path does not exist: {path}')
    paths = [path] if path.is_file() else sorted(path.rglob('*'))
    paths = [p for p in paths if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not paths:
        raise ValueError(f'No supported images found in {path}')
    return paths


def load_image(path, image_size=(320, 32)):
    """Return an RGB float tensor [3, H, W] in [0, 1]."""
    with Image.open(path) as image:
        image = image.convert('RGB').resize(
            (image_size[1], image_size[0]), Image.Resampling.LANCZOS
        )
        array = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


class BambooSlipsDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_size=(320, 32)):
        self.img_paths = image_paths(root_dir)
        self.image_size = tuple(image_size)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        path = self.img_paths[index]
        if self.transform is None:
            return load_image(path, self.image_size)
        with Image.open(path) as image:
            image = image.convert('RGB').resize(
                (self.image_size[1], self.image_size[0]), Image.Resampling.LANCZOS
            )
            return self.transform(image)
