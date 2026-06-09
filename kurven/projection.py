import numpy as np
from scipy.spatial.transform import Rotation as R


def rotate_xz(xyz, x_angle_deg, z_angle_deg):
    """Apply Rz then Rx (matches the original notebook's `rx.apply(rz.apply(xyz))`)."""
    rx = R.from_euler("x", x_angle_deg, degrees=True)
    rz = R.from_euler("z", z_angle_deg, degrees=True)
    return rx.apply(rz.apply(xyz))


def isometric_project(xyz, scale_factor=0.5, x_angle_deg=-55, z_angle_deg=-90):
    """Add isometric scale-shear in y, then rotate. Returns rotated 3D points.

    The shear `y' = y + scale_factor*x` produces the slight foreshortening that
    the gamma plate uses to spread points perpendicular to the viewer.
    """
    out = xyz.copy()
    out[:, 1] += scale_factor * out[:, 0]
    return rotate_xz(out, x_angle_deg, z_angle_deg)
