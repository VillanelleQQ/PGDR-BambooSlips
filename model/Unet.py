import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates


def get_norm(norm, num_channels, num_groups):
    if norm == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm == "bn":
        return nn.BatchNorm2d(num_channels)
    elif norm == "gn":
        return nn.GroupNorm(num_groups, num_channels)
    elif norm is None:
        return nn.Identity()
    else:
        raise ValueError("unknown normalization type")


class PositionalEmbedding(nn.Module):
    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ControlPointRegressor(nn.Module):
    def __init__(self, feature_channels, num_control_points, max_displacement=50.0):
        super().__init__()
        self.num_control_points = num_control_points
        self.max_displacement = max_displacement

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Linear(feature_channels, feature_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_channels // 2, feature_channels // 4),
            nn.ReLU(),
            nn.Linear(feature_channels // 4, num_control_points * 2)
        )

    def forward(self, features):
        global_features = self.global_pool(features).squeeze(-1).squeeze(-1)
        displacements = self.regressor(global_features)
        displacements = displacements.view(-1, self.num_control_points, 2)
        displacements = torch.tanh(displacements) * self.max_displacement
        return displacements


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)

    def forward(self, x, time_emb, y):
        if x.shape[2] % 2 == 1:
            raise ValueError("downsampling tensor height should be even")
        if x.shape[3] % 2 == 1:
            raise ValueError("downsampling tensor width should be even")
        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )

    def forward(self, x, time_emb, y):
        return self.upsample(x)


class EfficientAttentionBlock(nn.Module):
    def __init__(self, in_channels, norm="gn", num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        self.norm = get_norm(norm, in_channels, num_groups)
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.to_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        if h * w > 4096:
            return self._stripe_attention(x)
        else:
            return self._standard_attention(x)

    def _stripe_attention(self, x):
        b, c, h, w = x.shape
        max_stripe_pixels = 4096
        stripe_height = min(h, max_stripe_pixels // w)
        if h % stripe_height != 0:
            pad_h = stripe_height - (h % stripe_height)
            x_padded = F.pad(x, (0, 0, 0, pad_h))
            h_padded = h + pad_h
        else:
            x_padded = x
            h_padded = h
        num_stripes = h_padded // stripe_height
        output_stripes = []
        for i in range(num_stripes):
            start_h = i * stripe_height
            end_h = (i + 1) * stripe_height
            stripe = x_padded[:, :, start_h:end_h, :]
            stripe_attn = self._compute_attention(stripe)
            output_stripes.append(stripe_attn)
        output = torch.cat(output_stripes, dim=2)
        if h != h_padded:
            output = output[:, :, :h, :]
        return output

    def _compute_attention(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(self.norm(x))
        q, k, v = torch.split(qkv, self.in_channels, dim=1)
        q = q.view(b, c, h * w).transpose(1, 2)
        k = k.view(b, c, h * w).transpose(1, 2)
        v = v.view(b, c, h * w).transpose(1, 2)
        scale = c ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).view(b, c, h, w)
        return self.to_out(out) + x

    def _standard_attention(self, x):
        return self._compute_attention(x)


class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            dropout,
            time_emb_dim=None,
            num_classes=None,
            activation=F.relu,
            norm="gn",
            num_groups=32,
            use_attention=False,
    ):
        super().__init__()
        self.activation = activation
        self.norm_1 = get_norm(norm, in_channels, num_groups)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm_2 = get_norm(norm, out_channels, num_groups)
        self.conv_2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.time_bias = nn.Linear(time_emb_dim, out_channels) if time_emb_dim is not None else None
        self.class_bias = nn.Embedding(num_classes, out_channels) if num_classes is not None else None
        self.residual_connection = nn.Conv2d(in_channels, out_channels,
                                             1) if in_channels != out_channels else nn.Identity()
        self.attention = nn.Identity() if not use_attention else EfficientAttentionBlock(
            out_channels, norm, num_groups
        )

    def forward(self, x, time_emb=None, y=None):
        out = self.activation(self.norm_1(x))
        out = self.conv_1(out)
        if self.time_bias is not None:
            if time_emb is None:
                raise ValueError("time conditioning was specified but time_emb is not passed")
            out += self.time_bias(self.activation(time_emb))[:, :, None, None]
        if self.class_bias is not None:
            if y is None:
                raise ValueError("class conditioning was specified but y is not passed")
            out += self.class_bias(y)[:, :, None, None]
        out = self.activation(self.norm_2(out))
        out = self.conv_2(out) + self.residual_connection(x)
        out = self.attention(out)
        return out


class ControlPointUNet(nn.Module):
    def __init__(
            self,
            img_channels,
            base_channels,
            control_grid_size=(32, 4),
            channel_mults=(1, 2, 4, 8),
            num_res_blocks=2,
            time_emb_dim=None,
            time_emb_scale=1.0,
            num_classes=None,
            activation=F.relu,
            dropout=0.1,
            attention_resolutions=(),
            norm="gn",
            num_groups=32,
            initial_pad=0,
            max_displacement=50.0,
            image_size=(320, 32),
    ):
        super().__init__()
        self.activation = activation
        self.initial_pad = initial_pad
        self.max_displacement = max_displacement
        self.control_grid_size = control_grid_size
        self.image_size = image_size
        self.num_control_points = control_grid_size[0] * control_grid_size[1]
        self.register_buffer('control_points', self._setup_control_points())
        if len(attention_resolutions) == 0:
            attention_resolutions = self._auto_attention_resolutions(image_size, channel_mults)
        self.num_classes = num_classes
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(base_channels, time_emb_scale),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        ) if time_emb_dim is not None else None
        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlock(
                    now_channels,
                    out_channels,
                    dropout,
                    time_emb_dim=time_emb_dim,
                    num_classes=num_classes,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=i in attention_resolutions,
                ))
                now_channels = out_channels
                channels.append(now_channels)
            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(now_channels))
                channels.append(now_channels)
        self.mid = nn.ModuleList([
            ResidualBlock(
                now_channels,
                now_channels,
                dropout,
                time_emb_dim=time_emb_dim,
                num_classes=num_classes,
                activation=activation,
                norm=norm,
                num_groups=num_groups,
                use_attention=True,
            ),
            ResidualBlock(
                now_channels,
                now_channels,
                dropout,
                time_emb_dim=time_emb_dim,
                num_classes=num_classes,
                activation=activation,
                norm=norm,
                num_groups=num_groups,
                use_attention=False,
            ),
        ])
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.ups.append(ResidualBlock(
                    channels.pop() + now_channels,
                    out_channels,
                    dropout,
                    time_emb_dim=time_emb_dim,
                    num_classes=num_classes,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=i in attention_resolutions,
                ))
                now_channels = out_channels
            if i != 0:
                self.ups.append(Upsample(now_channels))
        assert len(channels) == 0
        self.out_norm = get_norm(norm, base_channels, num_groups)
        self.control_point_regressor = ControlPointRegressor(
            feature_channels=base_channels,
            num_control_points=self.num_control_points,
            max_displacement=max_displacement
        )

    def _auto_attention_resolutions(self, image_size, channel_mults):
        h, w = image_size
        attention_resolutions = []
        for i in range(len(channel_mults)):
            current_h = h // (2 ** i)
            current_w = w // (2 ** i)
            pixels = current_h * current_w
            if pixels <= 4096:
                attention_resolutions.append(i)
        return tuple(attention_resolutions)

    def _setup_control_points(self):
        img_h, img_w = self.image_size
        ny, nx = self.control_grid_size
        x_points = np.linspace(0, img_w - 1, nx)
        y_points = np.linspace(0, img_h - 1, ny)
        control_points = []
        for y in y_points:
            for x in x_points:
                control_points.append([x, y])
        return torch.tensor(control_points, dtype=torch.float32)

    def forward(self, x, deformation_step=None, y=None):
        ip = self.initial_pad
        if ip != 0:
            x = F.pad(x, (ip,) * 4)
        if self.time_mlp is not None:
            if deformation_step is None:
                raise ValueError("deformation step conditioning was specified but deformation_step is not passed")
            time_emb = self.time_mlp(deformation_step)
        else:
            time_emb = None
        if self.num_classes is not None and y is None:
            raise ValueError("class conditioning was specified but y is not passed")
        x = self.init_conv(x)
        skips = [x]
        for layer in self.downs:
            x = layer(x, time_emb, y)
            skips.append(x)
        for layer in self.mid:
            x = layer(x, time_emb, y)
        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x, time_emb, y)
        x = self.activation(self.out_norm(x))
        control_point_displacements = self.control_point_regressor(x)
        return control_point_displacements

    def predict_and_apply_deformation(self, deformed_image, deformation_step):
        predicted_displacements = self.forward(deformed_image, deformation_step)
        dense_displacement_field = self._control_points_to_dense_field(predicted_displacements)
        restored_image = self._apply_dense_displacement(deformed_image, dense_displacement_field)
        return restored_image, predicted_displacements

    def _control_points_to_dense_field(self, control_displacements):
        return self._control_points_to_dense_field_differentiable(control_displacements)

    def _control_points_to_dense_field_differentiable(self, control_displacements):
        batch_size = control_displacements.shape[0]
        device = control_displacements.device
        img_h, img_w = self.image_size
        ny, nx = self.control_grid_size
        disp_grid = control_displacements.view(batch_size, ny, nx, 2).permute(0, 3, 1, 2)
        dense_field = F.interpolate(
            disp_grid,
            size=(img_h, img_w),
            mode='bilinear',
            align_corners=True
        )
        return dense_field

    def _control_points_to_dense_field_original(self, control_displacements):
        batch_size = control_displacements.shape[0]
        img_h, img_w = self.image_size
        y_coords, x_coords = np.mgrid[0:img_h, 0:img_w]
        dense_fields = torch.zeros(batch_size, 2, img_h, img_w, device=control_displacements.device)
        for b in range(batch_size):
            control_points_np = self.control_points.cpu().numpy()
            displacements_np = control_displacements[b].detach().cpu().numpy()
            try:
                dx_dense = griddata(control_points_np, displacements_np[:, 0],
                                    (x_coords, y_coords), method='cubic', fill_value=0)
                dy_dense = griddata(control_points_np, displacements_np[:, 1],
                                    (x_coords, y_coords), method='cubic', fill_value=0)
            except:
                dx_dense = griddata(control_points_np, displacements_np[:, 0],
                                    (x_coords, y_coords), method='linear', fill_value=0)
                dy_dense = griddata(control_points_np, displacements_np[:, 1],
                                    (x_coords, y_coords), method='linear', fill_value=0)
            dense_fields[b, 0] = torch.from_numpy(dx_dense).float()
            dense_fields[b, 1] = torch.from_numpy(dy_dense).float()
        return dense_fields

    def _apply_dense_displacement(self, image, displacement_field):
        return self._apply_dense_displacement_differentiable(image, displacement_field)

    def _apply_dense_displacement_differentiable(self, image, displacement_field):
        batch_size, _, img_h, img_w = image.shape
        device = image.device
        y_grid = torch.linspace(-1, 1, img_h, device=device)
        x_grid = torch.linspace(-1, 1, img_w, device=device)
        grid_y, grid_x = torch.meshgrid(y_grid, x_grid, indexing='ij')
        disp_x = displacement_field[:, 0:1, :, :] / (img_w - 1) * 2
        disp_y = displacement_field[:, 1:2, :, :] / (img_h - 1) * 2
        sample_grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1) - disp_x
        sample_grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1) - disp_y
        sample_grid = torch.cat([sample_grid_x, sample_grid_y], dim=1).permute(0, 2, 3, 1)
        restored_image = F.grid_sample(
            image,
            sample_grid,
            mode='bilinear',
            padding_mode='reflection',
            align_corners=True
        )
        return restored_image

    def _apply_dense_displacement_original(self, image, displacement_field):
        batch_size, channels, img_h, img_w = image.shape
        y_coords, x_coords = np.mgrid[0:img_h, 0:img_w]
        restored_batch = torch.zeros_like(image)
        for b in range(batch_size):
            img_np = image[b].detach().cpu().numpy()
            disp_np = displacement_field[b].detach().cpu().numpy()
            map_x = x_coords - disp_np[0]
            map_y = y_coords - disp_np[1]
            restored_img = np.zeros_like(img_np)
            for c in range(channels):
                restored_img[c] = map_coordinates(
                    img_np[c], [map_y, map_x],
                    order=1, mode='reflect', prefilter=False
                )
            restored_batch[b] = torch.from_numpy(restored_img)
        return restored_batch.to(image.device)


def create_control_point_unet(img_channels=3, base_channels=64, control_grid_size=(32, 4)):
    return ControlPointUNet(
        img_channels=img_channels,
        base_channels=base_channels,
        control_grid_size=control_grid_size,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256,
        time_emb_scale=1.0,
        activation=F.relu,
        dropout=0.1,
        attention_resolutions=(),
        norm="gn",
        num_groups=32,
        max_displacement=50.0,
        image_size=(320, 32),
    )