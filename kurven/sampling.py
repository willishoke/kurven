import numpy as np


def sample_grid(x_min, x_max, y_min, y_max, nx, ny=None):
    if ny is None:
        ny = nx
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    return x, y


def complex_grid(real_min, real_max, imag_min, imag_max, n_real, n_imag=None):
    real, imag = sample_grid(real_min, real_max, imag_min, imag_max, n_real, n_imag)
    return real[:, None] + 1j * imag, real, imag
