# scripts/inspect_mpp_data.py — MPP 服务器数据预检工具
# 用途：核对 visiumhd_patch 中各 MPP 划分的患者 patch 数 + ssGSEA 列结构
# 运行环境：服务器 D:\AIPatho\qzs\pfmval_deploy_git/
#
# 用法:
#   cmd.exe /c "set PYTHONIOENCODING=utf-8 && python scripts/inspect_mpp_data.py"

import pandas as pd
import pathlib
import sys

ROOT = pathlib.Path(r"D:\AIPatho\Patch\visiumhd_patch")

CHECKS = [
    (3, "HYZ15040"), (3, "JFX"), (3, "LMZ12939"),
    (3, "TGC"), (3, "XSL"), (3, "ZHZ"),
    (2, "XZY"),
]

missing = []
for mpp_id, patient in CHECKS:
    d = ROOT / str(mpp_id) / patient
    if not d.exists():
        print(f"MISSING DIR: {d}")
        missing.append(str(d))
        continue

    pdir = d / "patch_images"
    if not pdir.exists():
        print(f"MISSING patch_images: {pdir}")
        missing.append(str(pdir))
        png = -1
    else:
        png = len(list(pdir.glob("*.png")))

    csv = d / f"{patient}_ssGSEA.csv"
    if not csv.exists():
        print(f"MISSING CSV: {csv}")
        missing.append(str(csv))
        continue

    df = pd.read_csv(csv)
    cols = list(df.columns)
    ncol = len(cols)
    print(f"MPP={mpp_id} {patient}: png={png}  csv_cols={cols[:3]}...  ncol={ncol}")

print(f"\nMISSING_TOTAL={len(missing)}")
if missing:
    for m in missing:
        print(f"  MISSING: {m}")
    sys.exit(1)
else:
    print("All checks passed.")
    sys.exit(0)
