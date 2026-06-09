import numpy as np

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = lambda x, **k: x


class ZBuffer:
    """Coord ↔ pixel mapping plus a depth buffer.

    Axis 0 of the buffer corresponds to `axis0_min..axis0_max`; axis 1 to
    `axis1_min..axis1_max`. The caller decides which projected coordinate is
    which — `ZBuffer` doesn't care about x/y semantics.
    """

    def __init__(
        self,
        axis0_min,
        axis0_max,
        axis1_min,
        axis1_max,
        shape,
        fill=-np.inf,
    ):
        self.buffer = np.full(shape, fill, dtype=float)
        self.lower = np.array([axis0_min, axis1_min], dtype=float)
        self.image_size = np.array(
            [axis0_max - axis0_min, axis1_max - axis1_min], dtype=float
        )
        self.buffer_size = np.array(shape, dtype=float)

    def coord_to_index(self, coord):
        # +0.01 nudge: keeps coord==lower from floor()ing to -1 under float jitter.
        return np.floor(0.01 + (self.buffer_size - 1) * (coord - self.lower) / self.image_size)

    def index_to_coord(self, index):
        return (index / (self.buffer_size - 1)) * self.image_size + self.lower


def surface_grid_mesh(coords_x, coords_y, z_values):
    """Build a regular triangulation of a 2D sample grid lifted to 3D.

    `coords_x`: shape (nx,) — x-axis coordinates
    `coords_y`: shape (ny,) — y-axis coordinates
    `z_values`: shape (ny, nx) — `z_values[i, j]` is the height at `(coords_x[j], coords_y[i])`

    Returns `(vertices, simplices)`:
      `vertices`: shape (ny*nx, 3) — vertex (i, j) at `(coords_x[j], coords_y[i], z_values[i, j])`
      `simplices`: shape (2*(ny-1)*(nx-1), 3) — two triangles per cell, indexing into `vertices`

    This is the right input to `rasterize_triangles` for a depth-buffer pass over
    the actual surface (as opposed to triangulating contour vertices, which has
    no meaningful relationship to the underlying surface).
    """
    coords_x = np.asarray(coords_x)
    coords_y = np.asarray(coords_y)
    z_values = np.asarray(z_values)
    ny, nx = z_values.shape
    if coords_x.shape[0] != nx or coords_y.shape[0] != ny:
        raise ValueError(
            f"z_values shape {z_values.shape} disagrees with "
            f"len(coords_x)={coords_x.shape[0]}, len(coords_y)={coords_y.shape[0]}"
        )

    X, Y = np.meshgrid(coords_x, coords_y)
    vertices = np.column_stack([X.ravel(), Y.ravel(), z_values.ravel()])

    i_idx, j_idx = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing="ij")
    i_idx = i_idx.ravel()
    j_idx = j_idx.ravel()
    v00 = i_idx * nx + j_idx
    v01 = i_idx * nx + (j_idx + 1)
    v10 = (i_idx + 1) * nx + j_idx
    v11 = (i_idx + 1) * nx + (j_idx + 1)

    n_cells = len(i_idx)
    simplices = np.empty((2 * n_cells, 3), dtype=np.int64)
    simplices[0::2, 0] = v00
    simplices[0::2, 1] = v01
    simplices[0::2, 2] = v10
    simplices[1::2, 0] = v01
    simplices[1::2, 1] = v11
    simplices[1::2, 2] = v10
    return vertices, simplices


def rasterize_triangles(
    zb,
    simplices,
    axis0_values,
    axis1_values,
    z_values,
    *,
    progress=False,
):
    """Z-buffer rasterize a triangle mesh into `zb`.

    `simplices` is (N, 3) int indices into the per-vertex arrays.
    `axis0_values`, `axis1_values`, `z_values` are (V,) per-vertex arrays.

    The inner loop computes barycentric weights directly from the three vertex
    positions instead of invoking `scipy.spatial.Delaunay()` per triangle, which
    is what the original notebook did and what made cell 14 the bottleneck.
    """
    bh, bw = int(zb.buffer_size[0]), int(zb.buffer_size[1])
    iterator = _tqdm(simplices) if progress else simplices

    for t in iterator:
        i0, i1, i2 = int(t[0]), int(t[1]), int(t[2])
        x0, x1, x2 = axis0_values[i0], axis0_values[i1], axis0_values[i2]
        y0, y1, y2 = axis1_values[i0], axis1_values[i1], axis1_values[i2]
        z0, z1, z2 = z_values[i0], z_values[i1], z_values[i2]

        bot_left = zb.coord_to_index(np.array([min(x0, x1, x2), min(y0, y1, y2)]))
        top_right = zb.coord_to_index(np.array([max(x0, x1, x2), max(y0, y1, y2)]))
        bottom = max(0, int(bot_left[0]))
        left = max(0, int(bot_left[1]))
        top = min(bh - 1, int(top_right[0]))
        right = min(bw - 1, int(top_right[1]))
        if top < bottom or right < left:
            continue

        x_ind, y_ind = np.meshgrid(
            np.arange(bottom, top + 1),
            np.arange(left, right + 1),
            indexing="ij",
        )
        xi_flat = x_ind.ravel()
        yi_flat = y_ind.ravel()
        xy = zb.index_to_coord(np.column_stack([xi_flat, yi_flat]))

        p0 = np.array([x0, y0])
        v0 = np.array([x1 - x0, y1 - y0])
        v1 = np.array([x2 - x0, y2 - y0])
        d00 = v0 @ v0
        d01 = v0 @ v1
        d11 = v1 @ v1
        denom = d00 * d11 - d01 * d01
        if denom == 0:
            continue

        v2 = xy - p0
        d20 = v2 @ v0
        d21 = v2 @ v1
        wv = (d11 * d20 - d01 * d21) / denom
        ww = (d00 * d21 - d01 * d20) / denom
        wu = 1.0 - wv - ww

        inside = (wu >= 0) & (wv >= 0) & (ww >= 0)
        if not inside.any():
            continue

        zs = wu * z0 + wv * z1 + ww * z2
        xi = xi_flat[inside]
        yi = yi_flat[inside]
        zi = zs[inside]
        # (xi, yi) pairs are unique within a single triangle's meshgrid, so a
        # plain assignment is safe and faster than np.maximum.at().
        zb.buffer[xi, yi] = np.maximum(zi, zb.buffer[xi, yi])


# Sentinel value representing "empty pixel" on the GPU path: GL has no -inf,
# and we want to round-trip the (h, w) buffer through an R32F texture, so we
# clear to this large negative value and remap back to -inf on read.
_GPU_EMPTY_SENTINEL = -1e30


def rasterize_triangles_gpu(zb, simplices, axis0_values, axis1_values, z_values):
    """GPU rasterization via moderngl. Same max-z semantics as the CPU loop.

    Renders the mesh into an R32F color texture with `GL_MAX` blend equation,
    then reads the texture back into `zb.buffer`. Existing values in `zb.buffer`
    are NOT preserved — the buffer is overwritten with the GPU result. (Matches
    the typical call pattern where `zb` is freshly constructed.)

    Requires `moderngl` (install with `pip install kurven[gpu]`).
    """
    try:
        import moderngl
    except ImportError as e:
        raise RuntimeError(
            "rasterize_triangles_gpu requires moderngl. "
            "Install with `pip install kurven[gpu]` or `uv sync --extra gpu`."
        ) from e

    h, w = int(zb.buffer_size[0]), int(zb.buffer_size[1])
    a0_min, a1_min = float(zb.lower[0]), float(zb.lower[1])
    a0_range, a1_range = float(zb.image_size[0]), float(zb.image_size[1])

    vbo_data = np.column_stack(
        [np.asarray(axis0_values), np.asarray(axis1_values), np.asarray(z_values)]
    ).astype(np.float32)
    ibo_data = np.asarray(simplices, dtype=np.int32)

    ctx = moderngl.create_standalone_context()
    try:
        prog = ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_vert;
                uniform vec2 a_min;
                uniform vec2 a_range;
                out float v_z;
                void main() {
                    // in_vert.x = axis 0 coord (mapped to NDC.y / framebuffer rows).
                    // in_vert.y = axis 1 coord (mapped to NDC.x / framebuffer cols).
                    float ndc_x = (in_vert.y - a_min.y) / a_range.y * 2.0 - 1.0;
                    float ndc_y = (in_vert.x - a_min.x) / a_range.x * 2.0 - 1.0;
                    gl_Position = vec4(ndc_x, ndc_y, 0.0, 1.0);
                    v_z = in_vert.z;
                }
            """,
            fragment_shader="""
                #version 330
                in float v_z;
                out float frag_z;
                void main() {
                    frag_z = v_z;
                }
            """,
        )
        prog["a_min"].value = (a0_min, a1_min)
        prog["a_range"].value = (a0_range, a1_range)

        vbo = ctx.buffer(vbo_data.tobytes())
        ibo = ctx.buffer(ibo_data.tobytes())
        vao = ctx.vertex_array(prog, [(vbo, "3f", "in_vert")], ibo)

        color_tex = ctx.texture((w, h), 1, dtype="f4")
        fbo = ctx.framebuffer(color_attachments=[color_tex])
        fbo.use()
        ctx.viewport = (0, 0, w, h)
        fbo.clear(color=(_GPU_EMPTY_SENTINEL, 0.0, 0.0, 0.0))

        ctx.enable(moderngl.BLEND)
        ctx.blend_equation = moderngl.MAX
        ctx.blend_func = (moderngl.ONE, moderngl.ONE)

        vao.render(moderngl.TRIANGLES)

        raw = fbo.read(components=1, dtype="f4")
        buf = np.frombuffer(raw, dtype=np.float32).reshape((h, w)).copy()
        buf[buf <= _GPU_EMPTY_SENTINEL + 1e20] = -np.inf
        zb.buffer[:] = buf
    finally:
        ctx.release()
