import torch


def get_mean_frequency_2d(coeff_2d: torch.Tensor):
    """Get the mean frequency of the 2D Fourier coefficients."""

    dim_u, dim_v = coeff_2d.shape
    max_freq = round(((dim_u - 1) ** 2 + (dim_v - 1) ** 2) ** (1 / 2))

    grid_i, grid_j = torch.meshgrid(
        torch.Tensor(range(dim_u)), torch.Tensor(range(dim_v)), indexing="ij"
    )
    bins = torch.round(torch.sqrt(grid_i**2 + grid_j**2)).to(torch.int)

    coeff_1d = torch.zeros(max_freq + 1)

    for i in range(max_freq + 1):
        coeff_1d[i] = coeff_2d[bins == i].sum()

    # Normalize
    coeff_1d[0] = 0  # ignore shift coming from 0 "Hz" frequency.
    coeff_1d /= coeff_1d.sum()

    # Compute mean frequency
    mean_freq = (coeff_1d * torch.Tensor(range(max_freq + 1))).sum() / max_freq

    return mean_freq.item()
