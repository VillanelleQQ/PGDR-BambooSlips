import torch
import torch.nn as nn
import torch.nn.functional as F
from forward import CreepDiffusionTrainer as ForwardTrainer
from forward import extract


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
        batch_size = x_0.shape[0]
        device = x_0.device

        t = torch.randint(1, self.T + 1, size=(batch_size,), device=device)

        x_t_batch = []
        target_incremental_displacements_batch = []

        for b in range(batch_size):
            x_t_single, incremental_displacement_at_t = self.forward_trainer.forward_step_by_step(
                x_0[b:b + 1], t[b].item()
            )

            target_inv_disp = -incremental_displacement_at_t.squeeze(0)

            x_t_batch.append(x_t_single)
            target_incremental_displacements_batch.append(target_inv_disp)

        x_t_batch = torch.cat(x_t_batch, dim=0)
        target_displacements_batch = torch.stack(target_incremental_displacements_batch, dim=0).to(device)

        predicted_displacements = self.model(x_t_batch, deformation_step=t)

        loss = F.mse_loss(predicted_displacements, target_displacements_batch)

        return loss


class CreepDiffusionSampler(nn.Module):
    def __init__(self, model, T=100):
        super().__init__()
        self.model = model
        self.T = T

    def forward(self, x_T, show_progress=True):
        x_t = x_T.clone()
        history = [x_t.clone()]

        time_steps = reversed(range(1, self.T + 1))

        for time_step in time_steps:
            t = x_t.new_ones([x_T.shape[0]], dtype=torch.long) * time_step

            with torch.no_grad():
                x_t_minus_1, _ = self.model.predict_and_apply_deformation(x_t, t)

            x_t = x_t_minus_1

            if time_step % (self.T // 10) == 0 or time_step == 1:
                history.append(x_t.clone())

        x_0 = torch.clamp(x_t, 0, 1)

        return x_0, history


def create_trainer(model, T=100, **physics_params):
    return CreepDiffusionTrainer(
        model=model, beta_1=1e-4, beta_T=0.02, T=T, **physics_params
    )


def create_sampler(model, T=100):
    return CreepDiffusionSampler(model=model, T=T)


def train_step(trainer, batch, optimizer):
    optimizer.zero_grad()
    loss = trainer(batch).mean()
    loss.backward()
    optimizer.step()
    return loss.item()


def restore(sampler, deformed_bamboo, show_progress=True):
    with torch.no_grad():
        return sampler(deformed_bamboo, show_progress)


def evaluate(original, restored):
    with torch.no_grad():
        mse = F.mse_loss(restored, original)
        psnr = -10 * torch.log10(mse + 1e-8)
    return {'mse': mse.item(), 'psnr': psnr.item()}