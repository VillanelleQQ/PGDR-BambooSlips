"""Save RGB tensors without importing unrelated vision/video modules."""
from PIL import Image


def save_image(tensor, path):
    tensor = tensor.detach().cpu()
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError('Expected one RGB image with shape [3, H, W] or [1, 3, H, W]')
    array = (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255 + 0.5).astype('uint8')
    Image.fromarray(array).save(path)
