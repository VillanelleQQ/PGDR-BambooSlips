"""Iterative restoration with the released displacement convention."""
import torch
from torch import nn


class CreepDiffusionSampler(nn.Module):
    def __init__(self, model, T=100):
        super().__init__()
        self.model = model
        if T < 1:
            raise ValueError("T must be positive")
        self.T = T

    def forward(self, x_T, show_progress=True, save_history=True):
        x_t = x_T.clone()
        history = [x_t.clone()] if save_history else []

        time_steps = reversed(range(1, self.T + 1))

        if show_progress:
            from tqdm import tqdm
            time_steps = tqdm(time_steps, total=self.T, desc="Restoring")

        for time_step in time_steps:
            t = x_t.new_ones([x_T.shape[0]], dtype=torch.long) * time_step

            with torch.no_grad():
                x_t_minus_1, _ = self.model.predict_and_apply_deformation(x_t, t)

            x_t = x_t_minus_1

            if save_history and (time_step % max(1, self.T // 10) == 0 or time_step == 1):
                history.append(x_t.clone())

        x_0 = torch.clamp(x_t, 0, 1)

        return x_0, history


