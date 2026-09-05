import numpy as np
from scipy.spatial.transform import Rotation as R


class Projection:
    """The shared 3D→2D camera for the landscape plates.

    Every example hand-rolled the same `project()` closure: an optional y-scale
    and z-clamp, an optional x-flip, an isometric y-shear by x, then Rz followed
    by Rx. This packages that as one configurable, callable object so the
    contours, the occluder mesh, and the scaffold ink all pass through one
    transform instead of a per-example closure.

    The three flipped examples are exact instances (apply() runs the identical
    operations in the identical order, so migration is bit-for-bit):

        recip:    Projection(shear=0.5,   x_angle=-55, flip_x=True)
        elliptic: Projection(shear=0.51,  x_angle=-63, flip_x=True)
        zeta:     Projection(shear=-0.18, x_angle=-79.5, flip_x=True,
                             y_scale=0.75, z_clamp=6.0)

        gamma:    Projection(shear=-0.5, x_angle=-55, flip_x=False)

    gamma was documented here as *not* being an instance of this convention. It
    is one exactly: it shears the other way (`+=` rather than `-=`, which is
    shear -0.5) and does not flip x. Over five hundred random points the
    inlined transform and this agree to zero, and the plate is pixel-identical
    through either.

    Pass a Projection anywhere a `project(xyz)` callable is expected — it is
    callable via __call__.
    """

    def __init__(self, *, shear, x_angle, z_angle=-90, flip_x=False,
                 y_scale=None, z_clamp=None):
        self.shear = shear
        self.x_angle = x_angle
        self.z_angle = z_angle
        self.flip_x = flip_x
        self.y_scale = y_scale
        self.z_clamp = z_clamp
        self.rx = R.from_euler("x", x_angle, degrees=True)
        self.rz = R.from_euler("z", z_angle, degrees=True)

    def apply(self, xyz):
        """Project (N, 3) (im, re, z) rows to rotated 3D coordinates. Take
        `[:, :2]` for the 2D plate."""
        out = np.array(xyz, dtype=float)
        if self.y_scale is not None:
            out[:, 1] *= self.y_scale
        if self.z_clamp is not None:
            out[:, 2] = np.minimum(self.z_clamp, out[:, 2])
        if self.flip_x:
            out[:, 0] *= -1
        out[:, 1] -= self.shear * out[:, 0]
        return self.rx.apply(self.rz.apply(out))

    def __call__(self, xyz):
        return self.apply(xyz)


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
