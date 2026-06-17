"""Phase timer for the example scripts: print per-phase wall-clock as each
phase closes, then a final percentage breakdown.

Pure stdout instrumentation — it never touches the rendered artifact. Replaces
the module-global `_TIMINGS`/`_LAST_TICK` + `_tick`/`_tick_done` pair that was
copy-pasted across gamma, zeta, and elliptic with a single per-instance timer.

    timer = PhaseTimer()
    timer.tick("load")    # opens "load"
    ...
    timer.tick("render")  # closes "load" (prints its dt), opens "render"
    ...
    timer.done()          # closes "render", prints the breakdown
"""

import time


class PhaseTimer:
    def __init__(self):
        self._timings = []
        self._last = None

    def tick(self, name):
        """Close the previous phase interval (printing its duration) and start
        a new phase `name`. The first call just opens a phase."""
        now = time.perf_counter()
        if self._last is not None:
            prev_name, prev_t = self._last
            dt = now - prev_t
            self._timings.append((prev_name, dt))
            print(f"  [{prev_name:>26s}] {dt:7.2f}s", flush=True)
        self._last = (name, now)

    def done(self):
        """Close the final phase and print the percentage breakdown + total."""
        self.tick("__end__")
        self._timings.pop()  # drop the dummy end marker
        total = sum(dt for _, dt in self._timings)
        print("  " + "-" * 40)
        for name, dt in self._timings:
            print(f"  [{name:>26s}] {dt:7.2f}s  {100*dt/total:4.1f}%")
        print(f"  [{'total':>26s}] {total:7.2f}s")
