"""Supervise control-point predictions with single-step inverse displacements."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from physics.forward import CreepForwardProcess as ForwardTrainer


class CreepDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T,
                 image_size=(320, 32), control_grid_size=(32, 4),
                 **physics_params):
        super().__init__()

        self.forward_trainer = ForwardTrainer(
            model=None,
            beta_1=beta_1,
            beta_T=beta_T,
            T=T,
            image_size=image_size,
            control_grid=control_grid_size,
            **physics_params
        )

        self.model = model
        self.physics_params = physics_params
        self.T = T

    def forward(self, x_0):
        return displacement_loss(self.model, self.forward_trainer, x_0, self.T)


def displacement_loss(model, forward_process, x_0, T):
    """Compute single-step control-point displacement MSE."""
    batch_size = x_0.shape[0]
    device = x_0.device

    t = torch.randint(1, T + 1, size=(batch_size,), device=device)

    x_t_batch = []
    target_incremental_displacements_batch = []

    for b in range(batch_size):
        x_t_single, incremental_displacement_at_t = forward_process.forward_step_by_step(
            x_0[b:b + 1], t[b].item()
        )

        target_inv_disp = -incremental_displacement_at_t.squeeze(0)

        x_t_batch.append(x_t_single)
        target_incremental_displacements_batch.append(target_inv_disp)

    x_t_batch = torch.cat(x_t_batch, dim=0)
    target_displacements_batch = torch.stack(target_incremental_displacements_batch, dim=0).to(device)

    predicted_displacements = model(x_t_batch, deformation_step=t)

    loss = F.mse_loss(predicted_displacements, target_displacements_batch)

    return loss


