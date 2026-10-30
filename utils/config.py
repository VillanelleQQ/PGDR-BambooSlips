"""Serializable model configuration shared by both entry points."""
MODEL_DEFAULTS = {
    'img_channels': 3,
    'base_channels': 64,
    'image_size': [320, 32],
    'control_grid_size': [32, 4],
    'channel_mults': [1, 2, 4, 8],
    'num_res_blocks': 2,
    'time_emb_dim': 256,
    'time_emb_scale': 1.0,
    'dropout': 0.2,
    'attention_resolutions': [1, 2],
    'norm': 'gn',
    'num_groups': 32,
    'max_displacement': 50.0,
}


def validate_model_config(config):
    height, width = config['image_size']
    ny, nx = config['control_grid_size']
    factor = 2 ** (len(config['channel_mults']) - 1)
    if min(height, width) <= 1 or height % factor or width % factor:
        raise ValueError(f'Image height/width must exceed 1 and be divisible by {factor}')
    if min(ny, nx) < 2:
        raise ValueError('The control grid needs at least two points per axis')
    if config['img_channels'] != 3:
        raise ValueError('The image loader produces RGB images; img_channels must be 3')
    if config['norm'] == 'gn':
        channels = [config['base_channels'] * m for m in config['channel_mults']]
        channels.append(config['base_channels'])
        if config['num_groups'] < 1 or any(c % config['num_groups'] for c in channels):
            raise ValueError('All feature widths must be divisible by positive num_groups')
    return config
