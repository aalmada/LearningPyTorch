import torch


def compute_accumulated_transmittance(betas):
    accumulated_transmittance = torch.cumprod(betas, 1)
    accumulated_transmittance[:, 0] = 1.
    return accumulated_transmittance


def rendering(model, rays_origins, rays_directions, tn, tf, number_bins=100, device='cpu'):

    t = torch.linspace(tn, tf, number_bins).to(device)  # [number_bins]
    delta = torch.cat(((t[1:] - t[:-1]), torch.tensor([1e10])))

    x = rays_origins.unsqueeze(1) + t.unsqueeze(0).unsqueeze(-1) * \
        rays_directions.unsqueeze(1)  # [number_rays, number_bins, 3]

    colors, density = model.intersect(x.reshape(-1, 3))

    # [number_rays, number_bins, 3]
    colors = colors.reshape((x.shape[0], number_bins, 3))
    density = density.reshape((x.shape[0], number_bins))

    # [number_rays, number_bins, 1]
    alpha = 1 - torch.exp(- density * delta.unsqueeze(0))
    T = compute_accumulated_transmittance(
        1 - alpha)  # [number_rays, number_bins, 1]

    c = (T.unsqueeze(-1) * alpha.unsqueeze(-1)
         * colors).sum(1)  # [number_rays, 3]

    return c
