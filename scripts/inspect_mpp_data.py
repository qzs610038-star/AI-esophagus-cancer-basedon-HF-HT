# scripts/inspect_mpp_data.py — MPP 服务器数据预检工具
# 用途：核对 visiumhd_patch 中各 MPP 划分的患者 patch 数 + ssGSEA 列结构
# 运行环境：服务器 D:\AIPatho\qzs\pfmval_deploy_git/
#
# 用法:
#   cmd.exe /c "set PYTHONIOENCODING=utf-8 && python scripts/inspect_mpp_data.py"

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from path_registry import get_registered_path

CHECKS = [
    (3, "HYZ15040"), (3, "JFX"), (3, "LMZ12939"),
    (3, "TGC"), (3, "XSL"), (3, "ZHZ"),
    (2, "XZY"),
]

def main() -> int:
    parser = argparse.ArgumentParser(description="MPP server data preflight (read-only)")
    parser.add_argument(
        "--mpp-root",
        default=str(get_registered_path("mpp_data_root")),
        help="MPP data root resolved from path id mpp_data_root by default",
    )
    args = parser.parse_args()
    root = pathlib.Path(args.mpp_root)

    missing = []
    for mpp_id, patient in CHECKS:
        directory = root / str(mpp_id) / patient
        if not directory.exists():
            print(f"MISSING DIR: {directory}")
            missing.append(str(directory))
            continue

        patches = directory / "patch_images"
        if not patches.exists():
            print(f"MISSING patch_images: {patches}")
            missing.append(str(patches))
            png_count = -1
        else:
            png_count = len(list(patches.glob("*.png")))

        label_csv = directory / f"{patient}_ssGSEA.csv"
        if not label_csv.exists():
            print(f"MISSING CSV: {label_csv}")
            missing.append(str(label_csv))
            continue

        frame = pd.read_csv(label_csv)
        columns = list(frame.columns)
        print(f"MPP={mpp_id} {patient}: png={png_count}  csv_cols={columns[:3]}...  ncol={len(columns)}")

    print(f"\nMISSING_TOTAL={len(missing)}")
    if missing:
        for item in missing:
            print(f"  MISSING: {item}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
