import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def extract(v, t, x_shape):
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def _apply_deformation_pytorch(image, control_points, control_displacements):
    batch_size, _, img_h, img_w = image.shape
    device = image.device

    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, img_h, device=device),
        torch.linspace(-1, 1, img_w, device=device),
        indexing='ij'
    )
    identity_grid = torch.stack((x_coords, y_coords), dim=2).unsqueeze(0).expand(batch_size, -1, -1, -1)

    norm_control_points = control_points.clone().to(device)
    norm_control_points[:, 0] = (norm_control_points[:, 0] / (img_w - 1)) * 2 - 1
    norm_control_points[:, 1] = (norm_control_points[:, 1] / (img_h - 1)) * 2 - 1

    norm_displacements = control_displacements.clone()
    norm_displacements[..., 0] = norm_displacements[..., 0] * (2.0 / (img_w - 1))
    norm_displacements[..., 1] = norm_displacements[..., 1] * (2.0 / (img_h - 1))

    pixel_coords = identity_grid.view(batch_size, -1, 2)
    dist = torch.cdist(pixel_coords, norm_control_points.unsqueeze(0).expand(batch_size, -1, -1))
    weights = 1.0 / (dist ** 2 + 1e-8)
    weights = weights / weights.sum(dim=2, keepdim=True)

    dense_displacement = torch.bmm(weights, norm_displacements)
    dense_displacement = dense_displacement.view(batch_size, img_h, img_w, 2)

    sampling_grid = identity_grid + dense_displacement

    deformed_image = F.grid_sample(image, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)
    return deformed_image


def create_gaussian_kernel(kernel_size=3, sigma=0.5):
    x = torch.arange(kernel_size) - kernel_size // 2
    y = torch.arange(kernel_size) - kernel_size // 2
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)


class CreepDeformationEngine:
    def __init__(self, image_size=(320, 32), control_grid=(32, 4)):
        self.img_h, self.img_w = image_size
        self.ny, self.nx = control_grid
        self.control_points = self._setup_control_points()
        self.gaussian_kernel = create_gaussian_kernel(kernel_size=3, sigma=0.5)
        self.fiber_orientations = torch.ones(self.ny * self.nx) * (np.pi / 2.0)

    def _setup_control_points(self):
        x_points = torch.linspace(0, self.img_w - 1, self.nx)
        y_points = torch.linspace(0, self.img_h - 1, self.ny)
        grid_y, grid_x = torch.meshgrid(y_points, x_points, indexing='ij')
        return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    def _compute_displacement_gradient(self, disp_field):
        disp_y = disp_field[:, 1].view(self.ny, self.nx)
        disp_x = disp_field[:, 0].view(self.ny, self.nx)

        grad_y_y = torch.zeros_like(disp_y)
        grad_y_y[1:-1, :] = (disp_y[2:, :] - disp_y[:-2, :]) / 2.0

        grad_x_x = torch.zeros_like(disp_x)
        grad_x_x[:, 1:-1] = (disp_x[:, 2:] - disp_x[:, :-2]) / 2.0

        gradient_magnitude = torch.sqrt(grad_y_y ** 2 + grad_x_x ** 2)
        return gradient_magnitude.flatten()

    def _update_fiber_orientations(self, stress_field, current_orientations,
                                   orientation_response_coeff=0.1,
                                   orientation_damping=0.8):
        orientations_grid = current_orientations.view(self.ny, self.nx)
        stress_grid = stress_field.view(self.ny, self.nx)

        angular_velocity = torch.zeros_like(orientations_grid)

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted_orient = torch.roll(orientations_grid, shifts=(dy, dx), dims=(0, 1))
            shifted_stress = torch.roll(stress_grid, shifts=(dy, dx), dims=(0, 1))

            angle_diff = shifted_orient - orientations_grid
            contribution = shifted_stress * torch.sin(angle_diff)
            angular_velocity += contribution

        angular_velocity = orientation_response_coeff * torch.tanh(angular_velocity / orientation_damping)
        new_orientations = current_orientations + angular_velocity.flatten()

        new_orientations = torch.fmod(new_orientations, np.pi)
        new_orientations[new_orientations < 0] += np.pi

        return new_orientations

    def _apply_stress_transfer(self, disp_field, stress_field, force_coupling_strength=0.3):
        disp_y_grid = disp_field[:, 1].view(1, 1, self.ny, self.nx)
        disp_x_grid = disp_field[:, 0].view(1, 1, self.ny, self.nx)
        stress_grid = stress_field.view(1, 1, self.ny, self.nx)

        gaussian_kernel = self.gaussian_kernel.to(disp_y_grid.device)
        stress_influence = F.conv2d(stress_grid, gaussian_kernel, padding='same')

        disp_y_transfer = disp_y_grid * (1.0 + force_coupling_strength * stress_influence)
        disp_x_transfer = disp_x_grid * (1.0 + force_coupling_strength * stress_influence)

        transferred_disp = torch.stack([disp_x_transfer.flatten(), disp_y_transfer.flatten()], dim=1)
        return transferred_disp

    def _compute_viscoelastic_time_factor(self, t, total_steps,
                                          elastic_modulus=0.6,
                                          viscosity_coeff=12.0,
                                          relaxation_time=None):
        if relaxation_time is None:
            tau = viscosity_coeff / elastic_modulus
        else:
            tau = relaxation_time

        normalized_t = t / total_steps
        time_factor = 1.0 - np.exp(-normalized_t * total_steps / tau)
        return time_factor

    def compute_bending_energy(self, displacement_field):
        disp_y = displacement_field[:, 1].view(self.ny, self.nx)
        curvature_approx = torch.zeros_like(disp_y)
        curvature_approx[:, 1:-1] = (disp_y[:, 2:] - 2 * disp_y[:, 1:-1] + disp_y[:, :-2])
        E_bend = torch.sum(curvature_approx ** 2)
        return E_bend.item()

    def compute_stretching_energy(self, displacement_field, reference_field=None):
        if reference_field is None:
            reference_field = torch.zeros_like(displacement_field)
        E_stretch = torch.sum((displacement_field - reference_field) ** 2)
        return E_stretch.item()

    def compute_total_energy(self, displacement_field):
        E_bend = self.compute_bending_energy(displacement_field)
        E_stretch = self.compute_stretching_energy(displacement_field)
        return E_bend + E_stretch

    def get_incremental_displacement(self, t, total_steps, cumulative_disp=None,
                                     fiber_orientations=None,
                                     fiber_elongation_factor=0.15,
                                     force_coupling_strength=0.3,
                                     moisture_diffusion_coeff=0.15,
                                     elastic_modulus=0.6,
                                     viscosity_coeff=12.0,
                                     relaxation_time=None,
                                     orientation_response_coeff=0.1,
                                     orientation_damping=0.8,
                                     **kwargs):
        num_points = self.control_points.shape[0]
        points = self.control_points
        center_y = self.img_h / 2.0

        if fiber_orientations is None:
            current_orientations = self.fiber_orientations
        else:
            current_orientations = fiber_orientations

        time_factor = self._compute_viscoelastic_time_factor(
            t, total_steps, elastic_modulus, viscosity_coeff, relaxation_time
        )

        dist_from_center_y = points[:, 1] - center_y
        disp_y_elong = time_factor * fiber_elongation_factor * dist_from_center_y
        disp_x_elong = torch.zeros_like(disp_y_elong)
        base_disp = torch.stack([disp_x_elong, disp_y_elong], dim=1)

        if cumulative_disp is not None:
            stress_proxy = self._compute_displacement_gradient(cumulative_disp)
        else:
            stress_proxy = torch.zeros(num_points)

        stress_transferred_disp = self._apply_stress_transfer(base_disp, stress_proxy, force_coupling_strength)

        moisture_stress_amp = 1.0 + moisture_diffusion_coeff * stress_proxy
        disp_y_coupled = stress_transferred_disp[:, 1] * moisture_stress_amp
        disp_x_coupled = stress_transferred_disp[:, 0] * moisture_stress_amp

        if cumulative_disp is not None:
            new_orientations = self._update_fiber_orientations(
                stress_proxy, current_orientations,
                orientation_response_coeff, orientation_damping
            )
            self.fiber_orientations = new_orientations
            orientation_weights = torch.abs(torch.sin(new_orientations))
            disp_y_coupled = disp_y_coupled * (1.0 + 0.2 * orientation_weights)

        noise = torch.randn_like(disp_y_coupled) * 0.05 * time_factor
        disp_y_final = disp_y_coupled + noise
        disp_x_final = disp_x_coupled

        return torch.stack([disp_x_final, disp_y_final], dim=1)


class CreepDiffusionTrainer(nn.Module):
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

    def forward(self, x_0):
        batch_size = x_0.shape[0]
        device = x_0.device

        t = torch.randint(1, self.T + 1, size=(batch_size,), device=device)
        x_t_batch = []
        target_displacements_batch = []

        for b in range(batch_size):
            target_t = t[b].item()

            self.creep_engine.fiber_orientations = torch.ones(
                self.creep_engine.ny * self.creep_engine.nx
            ) * (np.pi / 2.0)

            x_t_single, last_incremental_disp = self.forward_step_by_step(x_0[b:b + 1], target_t)

            x_t_batch.append(x_t_single)
            target_displacements_batch.append(-last_incremental_disp.squeeze(0))

        x_t_batch = torch.cat(x_t_batch, dim=0)
        target_displacements_batch = torch.stack(target_displacements_batch, dim=0).to(device)

        predicted_displacements = self.model(x_t_batch, t)
        loss = F.mse_loss(predicted_displacements, target_displacements_batch)

        return loss