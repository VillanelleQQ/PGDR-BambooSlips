"""Stepwise forward deformation used to synthesize training inputs."""
import torch
from torch import nn
from .creep import CreepDeformationEngine, _apply_deformation_pytorch


class CreepForwardProcess(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, image_size=(320, 32),
                 control_grid=(32, 4), **physics_params):
        super().__init__()
        self.model = model
        self.T = T
        self.image_size = image_size
        self.control_grid_size = control_grid
        self.physics_params = physics_params
        self.creep_engine = CreepDeformationEngine(image_size=image_size, control_grid=control_grid)

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward_step_by_step(self, x_0, target_t):
        x_current = x_0.clone()
        cumulative_disp = torch.zeros(self.creep_engine.control_points.shape[0], 2)
        fiber_orientations = self.creep_engine.fiber_orientations.clone()
        last_incremental_disp = None

        for t in range(1, target_t + 1):
            incremental_disp = self.creep_engine.get_incremental_displacement(
                t, self.T,
                cumulative_disp=cumulative_disp,
                fiber_orientations=fiber_orientations,
                **self.physics_params
            )

            incremental_disp_tensor = incremental_disp.to(x_current.device).unsqueeze(0)

            x_current = _apply_deformation_pytorch(
                x_current,
                self.creep_engine.control_points,
                incremental_disp_tensor
            )

            cumulative_disp = cumulative_disp + incremental_disp.cpu()
            fiber_orientations = self.creep_engine.fiber_orientations.clone()

            if t == target_t:
                last_incremental_disp = incremental_disp_tensor

        return x_current, last_incremental_disp

