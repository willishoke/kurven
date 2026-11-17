// The coverage measure of Swift's "preview: a parametric surface draws what
// its bake draws" (Sources/kurven-test/Surfaces.swift), in the page: the
// judged strokes rasterized into the preview's own pixels, against the
// pixels the preview darkened. Injected by the browser lane as a classic
// script; it defines `kurvenCoverage` and nothing else.
//
// - A preview pixel is a stroke where its luminance, 299 r + 587 g + 114 b,
//   is under 224,000: any visible mark counts, since a stroke narrower than
//   a pixel is drawn lighter, not narrower.
// - The judged strokes are rasterized along each segment's centreline at a
//   quarter of a pixel, each sample marking the pixel `DepthFrame.index`
//   puts it in.
// - Recall: of the judged pixels, the share with a preview stroke within one
//   pixel. Precision: of the preview's stroke pixels, the share with a judged
//   pixel within two. Swift holds both above 0.99, with more than 5,000
//   judged pixels.
// - The control: the same frame's curves rasterized with no visibility test
//   at all. If the measure could not tell that from the real thing it would
//   prove nothing; Swift asks that over a fifth of its pixels have no judged
//   stroke near.

globalThis.kurvenCoverage = (() => {
  function strokeMask(pixels) {
    const { width, height, order, data } = pixels;
    const mask = new Uint8Array(width * height);
    const [ri, bi] = order === "bgra" ? [2, 0] : [0, 2];
    for (let k = 0; k < width * height; k++) {
      const lum = 299 * data[4 * k + ri] + 587 * data[4 * k + 1] + 114 * data[4 * k + bi];
      mask[k] = lum < 224000 ? 1 : 0;
    }
    return mask;
  }

  function index(frame, x, y) {
    const row = Math.floor(frame.nudge + ((frame.rows - 1) * (y - frame.axis0[0])) / (frame.axis0[1] - frame.axis0[0]));
    const col = Math.floor(frame.nudge + ((frame.cols - 1) * (x - frame.axis1[0])) / (frame.axis1[1] - frame.axis1[0]));
    return [row, col];
  }

  function unitsPerPixel(frame) {
    return Math.max(
      Math.abs(frame.axis1[1] - frame.axis1[0]) / Math.max(frame.cols - 1, 1),
      Math.abs(frame.axis0[1] - frame.axis0[0]) / Math.max(frame.rows - 1, 1),
    );
  }

  /** Mark the pixels of polylines given as view-plane points: `paths` is a
   * list of `{ x(k), y(k), offsets }`. */
  function rasterize(frame, paths) {
    const { rows, cols } = frame;
    const mask = new Uint8Array(rows * cols);
    const upp = unitsPerPixel(frame);
    for (const p of paths) {
      for (let i = 0; i + 1 < p.offsets.length; i++) {
        for (let k = p.offsets[i]; k + 1 < p.offsets[i + 1]; k++) {
          const ax = p.x(k), ay = p.y(k), bx = p.x(k + 1), by = p.y(k + 1);
          const steps = Math.max(Math.trunc((Math.hypot(bx - ax, by - ay) / upp) * 4), 1);
          for (let j = 0; j <= steps; j++) {
            const t = j / steps;
            const [r, c] = index(frame, ax + (bx - ax) * t, ay + (by - ay) * t);
            if (r >= 0 && r < rows && c >= 0 && c < cols) mask[r * cols + c] = 1;
          }
        }
      }
    }
    return mask;
  }

  /** The judged strokes of a `bake` reply: page points are view x and y.
   * `only`, when given, is the one stroke layer to rasterize. */
  function judgedMask(frame, reply, arrays, only) {
    const layers = reply.strokes.map((_, i) => i).filter((i) => only === undefined || i === only);
    const paths = layers.map((i) => {
      const pts = arrays[`stroke.${i}.points`];
      return { offsets: arrays[`stroke.${i}.offsets`], x: (k) => pts[2 * k], y: (k) => pts[2 * k + 1] };
    });
    return rasterize(frame, paths);
  }

  /** A `frame` reply's world-space curves, through the view matrix (column
   * by column), drawn whole: the control. */
  function unclippedMask(frame, view, reply, arrays) {
    const project = (v, k, row) => view[row] * v[3 * k] + view[4 + row] * v[3 * k + 1] + view[8 + row] * v[3 * k + 2] + view[12 + row];
    const paths = reply.layers
      .map((l, i) => ({ l, v: arrays[`layer.${i}.vertices`], o: arrays[`layer.${i}.offsets`] }))
      .filter((p) => p.l.visible && p.v && p.o)
      .map((p) => ({ offsets: p.o, x: (k) => project(p.v, k, 0), y: (k) => project(p.v, k, 1) }));
    return rasterize(frame, paths);
  }

  function near(mask, k, r, width, height) {
    const row = Math.floor(k / width), col = k % width;
    for (let dr = -r; dr <= r; dr++) {
      for (let dc = -r; dc <= r; dc++) {
        const rr = row + dr, cc = col + dc;
        if (rr >= 0 && rr < height && cc >= 0 && cc < width && mask[rr * width + cc]) return true;
      }
    }
    return false;
  }

  function measure(preview, judged, width, height) {
    let baked = 0, shown = 0, stroked = 0, backed = 0;
    for (let k = 0; k < width * height; k++) {
      if (judged[k]) {
        baked++;
        if (near(preview, k, 1, width, height)) shown++;
      }
      if (preview[k]) {
        stroked++;
        if (near(judged, k, 2, width, height)) backed++;
      }
    }
    return { baked, shown, recall: shown / Math.max(baked, 1), stroked, backed, precision: backed / Math.max(stroked, 1) };
  }

  /** Of a mask's pixels, the share with no judged pixel within two. */
  function stray(mask, judged, width, height) {
    let all = 0, far = 0;
    for (let k = 0; k < width * height; k++) {
      if (!mask[k]) continue;
      all++;
      if (!near(judged, k, 2, width, height)) far++;
    }
    return { pixels: all, stray: far / Math.max(all, 1) };
  }

  /** Swift's "metal: the GPU coordinate pass is the CPU rasterizer's":
   * coverage and depth differ only on edges -- of the drawing, or of a sheet
   * in front of another, where some neighbouring pixel of the CPU image is
   * empty or far away -- and on under half a percent of the pixels; and where
   * the two agree, so do the coordinates, to within a grid cell (the short
   * way round a periodic axis), which is the front-most point the CPU image's
   * holds. `surface` is the resources' `{ cells, samples, spacing }`. */
  function compareImages(gpu, cpu, rows, cols, span, surface) {
    const isCovered = (img, k) => img.depth[k] > -1e29;
    const edgeAt = (k) => {
      const row = Math.floor(k / cols), col = k % cols;
      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          const rr = row + dr, cc = col + dc;
          if (rr < 0 || rr >= rows || cc < 0 || cc >= cols) return true;
          const n = rr * cols + cc;
          if (!isCovered(cpu, n) || Math.abs(cpu.depth[n] - cpu.depth[k]) > 0.05 * span) return true;
        }
      }
      return false;
    };
    const delta = (d, axis) => {
      const period = surface.spacing[axis] * surface.cells[axis];
      return surface.cells[axis] === surface.samples[axis] ? d - period * Math.round(d / period) : d;
    };
    let covered = 0, strays = 0, onEdge = 0, worstDepth = 0, checked = 0, coordWrong = 0, worstCoord = 0;
    for (let k = 0; k < rows * cols; k++) {
      const a = isCovered(gpu, k), b = isCovered(cpu, k);
      if (a || b) covered++;
      const differs = a !== b || (a && Math.abs(gpu.depth[k] - cpu.depth[k]) > 1e-4 * span);
      if (a && b && !differs) {
        worstDepth = Math.max(worstDepth, Math.abs(gpu.depth[k] - cpu.depth[k]));
        if (!edgeAt(k)) {
          checked++;
          let off = 0;
          for (const axis of [0, 1]) {
            const d = Math.abs(delta(gpu.coords[2 * k + axis] - cpu.coords[2 * k + axis], axis));
            off = Math.max(off, d / surface.spacing[axis]);
          }
          worstCoord = Math.max(worstCoord, off);
          if (!(off <= 1)) coordWrong++;
        }
      }
      if (!differs) continue;
      strays++;
      if (edgeAt(k)) onEdge++;
    }
    return { covered, strays, onEdge, worstDepth, checked, coordWrong, worstCoord };
  }

  return { strokeMask, judgedMask, unclippedMask, measure, stray, compareImages };
})();
