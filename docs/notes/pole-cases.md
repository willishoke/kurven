# What went wrong at a pole

*A record, moved here from the README on 2026-10-11. It was written while
visibility was decided by a depth buffer with a margin (June to September
2026); since then every heightfield line is judged by facing and an exact
march on f (see [architecture.md](../architecture.md#6-judge)), and several
of the fixes below were replaced by that rule rather than kept. The commits
are the record of how each case was found.*

Every hard case in a plate is a boundary condition: a pole, a cap, a cut face,
a steep flank. Each of these was a visible defect with a commit behind it, and
each is a picture the pipeline can draw both ways.

- **A pole is a needle and a floor.** The cap says where to cut, and gamma's
  caps vary by band of Re so each spire's plateau reads from above. The cut
  faces and the plateaus are hatched, which is the part every published plate
  wrote by hand (see *[Truncation](../landscapes.md#truncation-and-the-hatching-that-goes-with-it)*).
- **The rim is not a contour.** Under band caps the cap outline is the zero set
  of |f| − cap(x), which is a contour of nothing else; it has its own layer,
  and the magnitude level at exactly the cap is dropped so the line is not
  drawn twice (`87ff162`).
- **A plateau is blank but for its hatching.** Phase contours are lifted to
  the *unclamped* magnitude and cut the same way, so they end at the rim
  rather than converging on the pole across the cap (`87ff162`).
- **The occluder must not bridge a notch.** The first occluder was a Delaunay
  triangulation of the contour vertices, which triangulates the convex hull
  and so spanned the elliptic plate's cutout with phantom triangles that hid
  the front face. A meshed heightfield with wall curtains replaced it, and
  `MAX` blending tolerates the tile seams, so the cutout is handled by
  omission (`26f3e05`).
- **Ink on a steep flank flickers.** The preview tests a line's depth where it
  crosses a pixel against the surface's depth at the pixel's centre, up to
  half a pixel apart; on a surface steep in view that is more depth than the
  margin. The margin is scaled by the surface's own slope, as shadow maps bias
  by slope, and the crawl goes from 2.2% of ink per frame to 0.3% (`c81ff55`).
- **The last vertex under the cap.** A contour kept below the cap used to end
  at its last sample under it, up to a sample's rise short of the rim. The
  crossing with the cap plane is now solved on the lifted heights and the run
  ends on the rim at exactly the cap's height (`bd605b1`).
- **The cell that straddles the cap.** Clamping heights at the lattice
  vertices and interpolating between them draws a cell with one corner over
  the cap as a slope from the cap to the low corner, cutting off the corner
  where the wall meets the cap by up to the cell's rise, half a world unit on
  a pole's flank. The cells a cap crosses are drawn as min(interpolated, cap),
  exactly (`52d62b7`, `18fe661`).
- **Back-face ink inside the margin.** Where a flank turns away, its back
  lies within the margin of its front for a stretch that zoom magnifies, and
  the other family of contours showed through as ticks along the silhouette.
  The facing test hides it before the depth test is asked (`b1bbf95`).
- **What remains.** At a needle a few cells wide a contour genuinely hooks
  round its edge: 0.35 px at plate resolution, and correct.
