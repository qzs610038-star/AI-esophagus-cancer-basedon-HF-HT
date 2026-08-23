import importlib.util
from pathlib import Path
import numpy as np

MODULE = Path(__file__).with_name("run_recalculation.py")
spec = importlib.util.spec_from_file_location("recalc", MODULE)
recalc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recalc)


def main():
    x = np.arange(49, dtype=float)
    assert abs(recalc.pcc(x, x) - 1) < 1e-12
    assert abs(recalc.ccc(x, x) - 1) < 1e-12
    assert recalc.pcc(x, x + 5) > 0.999999
    assert recalc.ccc(x, x + 5) < 1
    coords = np.arange(7)
    xx, yy = np.meshgrid(coords, coords)
    score, occupancy, windows, _, _ = recalc.local_ssim(x, x, xx.ravel(), yy.ravel())
    assert abs(score - 1) < 1e-12 and occupancy == 1 and windows == 1
    assert recalc.parse_xy("patch_x123_y456") == (123, 456)
    print("minimal metric tests: PASS")


if __name__ == "__main__":
    main()
