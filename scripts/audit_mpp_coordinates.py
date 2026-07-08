#!/usr/bin/env python3
"""
scripts/audit_mpp_coordinates.py — MPP1-5 统一标准重跑前置审计

按方案 §十二 P0-2 / P1-1 / P0-3 排查项，在服务器只读扫描：
  1. raw ssGSEA 标签：列结构、barcode 格式、30 通路列名跨 MPP 一致性 (P0-2)
  2. patch 坐标分布：坐标范围、unique 数、相邻 patch 距离直方图 (验证步长 224 / 112) (P1-1)
  3. 现有特征缓存：partner 缓存 (MPP1_UNI/MPP4_UNI) train/val 子目录 .pt 分布；
     扁平缓存 (mpp_uni2h_cache/3, mpp_uni2h_cache/2/XZY) .pt 数 (P0-3)
  4. patch 数 vs 路径索引 patch_images 表核对

只读文件名 + CSV 头，不加载 .pt 内容，不加载图像，无需 GPU。服务器秒级跑完。

输出:
  - 控制台汇总（人类可读）
  - audit_mpp_coordinates_report.csv（机器可读，一行一审计单元）
  - split_candidate_input.csv（每 MPP×患者：坐标范围、unique 数、patch 数、
    建议步长，供 generate_standard_splits.py 选 block_size 用）

用法（服务器）:
    cd D:\\AIPatho\\qzs\\pfmval_deploy_git
    "C:\\Users\\AIPatho1\\pfmval_env\\Scripts\\python.exe" scripts/audit_mpp_coordinates.py
    # 或指定根目录
    "C:\\Users\\AIPatho1\\pfmval_env\\Scripts\\python.exe" scripts/audit_mpp_coordinates.py \\
        --mpp-root D:\\AIPatho\\Patch\\visiumhd_patch \\
        --cache-root D:\\AIPatho\\qzs\\pfmval_deploy_git\\uni2h_cache \\
        --flat-cache-root D:\\AIPatho\\qzs\\pfmval_deploy_git\\mpp_uni2h_cache

本地小样本验证（无需 GPU，跳过 缓存审计）:
    "C:\\Program Files\\Python313\\python.exe" scripts/audit_mpp_coordinates.py --skip-cache
"""

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# ── 常量 ──
DEFAULT_MPP_ROOT = r"D:\AIPatho\Patch\visiumhd_patch"
DEFAULT_CACHE_ROOT = r"D:\AIPatho\qzs\pfmval_deploy_git\uni2h_cache"  # partner: MPP{N}_UNI/
DEFAULT_FLAT_CACHE_ROOT = r"D:\AIPatho\qzs\pfmval_deploy_git\mpp_uni2h_cache"  # 扁平: {N}/{patient}/

# 方案 §二 固定的六例非 XZY 训练患者 + 外部测试
TRAIN_PATIENTS = ["HYZ15040", "JFX", "LMZ12939", "TGC", "XSL", "ZHZ"]
EXTERNAL_PATIENT = "XZY"
MPP_IDS = [1, 2, 3, 4, 5]

# 路径索引 §MPP 数据根 patch_images 数量表（用于 patch 数核对）
EXPECTED_PATCH_COUNTS = {
    1: {"HYZ15040": 3071, "JFX": 7788, "LMZ12939": 7275, "TGC": 1116, "XSL": 1526, "XZY": 1039, "ZHZ": 1074},
    2: {"HYZ15040": 1454, "JFX": 2029, "LMZ12939": 3351, "TGC": 1116, "XSL": 1526, "XZY": 1039, "ZHZ": 1074},
    3: {"HYZ15040": 5807, "JFX": 8116, "LMZ12939": 13376, "TGC": 4476, "XSL": 6093, "XZY": 4139, "ZHZ": 4257},
    4: {"HYZ15040": 429, "JFX": 541, "LMZ12939": 889, "TGC": 308, "XSL": 396, "XZY": 269, "ZHZ": 282},
    5: {"HYZ15040": 1722, "JFX": 2162, "LMZ12939": 3549, "TGC": 1206, "XSL": 1599, "XZY": 1082, "ZHZ": 1132},
}

# 30 通路列名（参考组，来自 JFX raw ssGSEA CSV）
REF_PATHWAYS = [
    "tls", "tgfb", "emt", "hypoxia", "mhc", "icp", "ifng", "toxic", "Glycolysis",
    "Inflammatory_Response", "IL6_JAK_STAT3", "P53_Pathway", "DNA_Damage_Response",
    "Complement", "Coagulation", "Oxidative_Phosphorylation", "Reactive_Oxygen_Species",
    "Wound_Healing", "Fibrosis", "MYC_Targets", "E2F_Targets", "G2M_Checkpoint",
    "Mitotic_Spindle", "Unfolded_Protein_Response", "mTOR_Signaling", "Interferon_Alpha",
    "Angiogenesis", "Apoptosis", "TNF_Signaling", "ECM_Organization",
]

COORD_RE = re.compile(r'x(\d+)_y(\d+)')


def parse_xy(stem: str):
    """从 'patch_x10192_y10192' 解析 (x, y)；无匹配返回 (None, None)。"""
    m = COORD_RE.search(stem)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def banner(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ═══════════════════════════════════════════════════════════════
# 1. raw ssGSEA 标签审计
# ═══════════════════════════════════════════════════════════════

def audit_raw_labels(mpp_root: Path, report_rows: list):
    """审计每个 MPP×患者的 raw ssGSEA CSV：列结构、barcode 格式、通路列一致性。"""
    banner("1. raw ssGSEA 标签审计 (P0-2)")
    all_pathways_consistent = True
    all_barcode_parseable = True

    for mpp_id in MPP_IDS:
        for patient in TRAIN_PATIENTS + [EXTERNAL_PATIENT]:
            csv_path = mpp_root / str(mpp_id) / patient / f"{patient}_ssGSEA.csv"
            row = {
                "audit_type": "raw_label",
                "mpp_id": mpp_id,
                "patient": patient,
                "path": str(csv_path),
                "exists": csv_path.exists(),
            }
            if not csv_path.exists():
                row["status"] = "MISSING"
                row["note"] = "raw ssGSEA CSV 不存在"
                report_rows.append(row)
                print(f"  [MISSING] MPP-{mpp_id}/{patient}: {csv_path}")
                continue

            try:
                df = pd.read_csv(csv_path, nrows=5)
            except Exception as e:
                row["status"] = "READ_ERROR"
                row["note"] = f"读取失败: {e}"
                report_rows.append(row)
                print(f"  [ERROR]  MPP-{mpp_id}/{patient}: 读取失败 {e}")
                continue

            cols = list(df.columns)
            n_cols = len(cols)
            first_col = cols[0]
            pathways = cols[1:]

            # 通路列一致性
            pathways_match = (pathways == REF_PATHWAYS)
            if not pathways_match:
                all_pathways_consistent = False

            # barcode 格式可解析性
            barcode_sample = str(df[first_col].iloc[0]) if n_cols > 0 else ""
            x, y = parse_xy(barcode_sample)
            barcode_parseable = (x is not None)
            if not barcode_parseable:
                all_barcode_parseable = False

            # 完整行数（不读全部，只读前5行 sample；后续步骤会读全量）
            row.update({
                "n_cols": n_cols,
                "first_col": first_col,
                "n_pathways": len(pathways),
                "pathways_match_ref": pathways_match,
                "barcode_sample": barcode_sample,
                "barcode_parseable": barcode_parseable,
                "status": "OK" if (n_cols == 31 and pathways_match and barcode_parseable) else "WARN",
            })
            report_rows.append(row)

            status_icon = "OK" if row["status"] == "OK" else "WARN"
            print(f"  [{status_icon}] MPP-{mpp_id}/{patient}: "
                  f"cols={n_cols}, first='{first_col}', pathways_match={pathways_match}, "
                  f"barcode='{barcode_sample}' parseable={barcode_parseable}")

    print(f"\n  汇总: 通路列跨MPP一致={all_pathways_consistent}, "
          f"barcode全部可解析={all_barcode_parseable}")
    if not all_pathways_consistent:
        print("  ⚠️  通路列名跨 MPP 不一致，z-score 重建脚本需做列名对齐")
    if not all_barcode_parseable:
        print("  ⚠️  部分 barcode 不可解析坐标，需检查 CSV 首列格式")
    return all_pathways_consistent and all_barcode_parseable


# ═══════════════════════════════════════════════════════════════
# 2. patch 坐标分布 + 步长审计
# ═══════════════════════════════════════════════════════════════

def audit_patch_coordinates(mpp_root: Path, report_rows: list, candidate_rows: list):
    """审计每 MPP×患者 patch_images 坐标分布、步长直方图、patch 数核对。"""
    banner("2. patch 坐标分布 + 步长审计 (P1-1)")

    for mpp_id in MPP_IDS:
        for patient in TRAIN_PATIENTS + [EXTERNAL_PATIENT]:
            patch_dir = mpp_root / str(mpp_id) / patient / "patch_images"
            row = {
                "audit_type": "patch_coord",
                "mpp_id": mpp_id,
                "patient": patient,
                "path": str(patch_dir),
                "exists": patch_dir.exists(),
            }
            if not patch_dir.exists():
                row["status"] = "MISSING"
                row["note"] = "patch_images 目录不存在"
                report_rows.append(row)
                print(f"  [MISSING] MPP-{mpp_id}/{patient}: {patch_dir}")
                continue

            # 只列文件名，不读图像内容
            stems = [p.stem for p in patch_dir.iterdir() if p.suffix.lower() == ".png"]
            n_patches = len(stems)
            expected = EXPECTED_PATCH_COUNTS.get(mpp_id, {}).get(patient, -1)
            count_match = (n_patches == expected)

            # 解析坐标
            coords = []
            unparseable = 0
            for s in stems:
                x, y = parse_xy(s)
                if x is not None:
                    coords.append((x, y))
                else:
                    unparseable += 1

            if not coords:
                row.update({
                    "n_patches": n_patches,
                    "expected": expected,
                    "count_match": count_match,
                    "unparseable": unparseable,
                    "status": "ERROR",
                    "note": "无坐标可解析",
                })
                report_rows.append(row)
                print(f"  [ERROR]  MPP-{mpp_id}/{patient}: 无坐标可解析")
                continue

            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            unique_coords = len(set(coords))
            x_range = (min(xs), max(xs))
            y_range = (min(ys), max(ys))

            # 步长直方图：按 x 排序后相邻差、y 排序后相邻差
            sorted_x = sorted(set(xs))
            sorted_y = sorted(set(ys))
            dx_counts = Counter(sorted_x[i + 1] - sorted_x[i] for i in range(len(sorted_x) - 1))
            dy_counts = Counter(sorted_y[i + 1] - sorted_y[i] for i in range(len(sorted_y) - 1))

            # 主步长（出现次数最多的差值）
            main_dx = dx_counts.most_common(1)[0][0] if dx_counts else 0
            main_dy = dy_counts.most_common(1)[0][0] if dy_counts else 0

            # 步长判定：50% overlap → 主步长≈112；100% → 主步长≈224
            stride_guess = "100%_stride" if main_dx >= 200 else "50%_stride" if main_dx >= 100 else "unknown"

            row.update({
                "n_patches": n_patches,
                "expected": expected,
                "count_match": count_match,
                "unique_coords": unique_coords,
                "x_range": f"{x_range[0]}-{x_range[1]}",
                "y_range": f"{y_range[0]}-{y_range[1]}",
                "main_dx": main_dx,
                "main_dy": main_dy,
                "stride_guess": stride_guess,
                "unparseable": unparseable,
                "status": "OK" if (count_match and unparseable == 0) else "WARN",
            })
            report_rows.append(row)
            report_rows.append({
                "audit_type": "stride_hist",
                "mpp_id": mpp_id,
                "patient": patient,
                "dx_hist_top3": str(dict(dx_counts.most_common(3))),
                "dy_hist_top3": str(dict(dy_counts.most_common(3))),
            })

            # split_candidate_input: 供 block_size 选择
            candidate_rows.append({
                "mpp_id": mpp_id,
                "patient": patient,
                "n_patches": n_patches,
                "unique_coords": unique_coords,
                "x_min": x_range[0], "x_max": x_range[1],
                "y_min": y_range[0], "y_max": y_range[1],
                "main_dx": main_dx, "main_dy": main_dy,
                "stride_guess": stride_guess,
            })

            status_icon = "OK" if row["status"] == "OK" else "WARN"
            count_note = "" if count_match else f" (期望{expected}, 不匹配!)"
            print(f"  [{status_icon}] MPP-{mpp_id}/{patient}: "
                  f"n={n_patches}{count_note}, unique={unique_coords}, "
                  f"dx={main_dx}, dy={main_dy}, stride={stride_guess}")

    print(f"\n  步长判定: main_dx>=200 → 100%步长(无需 embargo); "
          f"100<=main_dx<200 → 50%步长(需 embargo)")


# ═══════════════════════════════════════════════════════════════
# 3. 现有特征缓存审计
# ═══════════════════════════════════════════════════════════════

def audit_partner_cache(cache_root: Path, mpp_ids: list, report_rows: list):
    """审计 partner 缓存 MPP{N}_UNI/{patient}/{train|val}/*.pt 子目录分布 (P0-3)。"""
    banner(f"3a. partner 缓存审计 (MPP{{N}}_UNI/) — P0-3")
    print(f"  cache_root: {cache_root}")

    for mpp_id in mpp_ids:
        mpp_dir = cache_root / f"MPP{mpp_id}_UNI"
        if not mpp_dir.exists():
            print(f"  [MISSING] {mpp_dir}")
            report_rows.append({
                "audit_type": "partner_cache",
                "mpp_id": mpp_id,
                "patient": "ALL",
                "path": str(mpp_dir),
                "exists": False,
                "status": "MISSING",
            })
            continue

        for patient in TRAIN_PATIENTS + [EXTERNAL_PATIENT]:
            p_dir = mpp_dir / patient
            if not p_dir.exists():
                report_rows.append({
                    "audit_type": "partner_cache",
                    "mpp_id": mpp_id, "patient": patient,
                    "path": str(p_dir), "exists": False, "status": "MISSING",
                })
                print(f"  [MISSING] MPP{mpp_id}_UNI/{patient}")
                continue

            train_n = len(list((p_dir / "train").glob("*.pt"))) if (p_dir / "train").exists() else 0
            val_n = len(list((p_dir / "val").glob("*.pt"))) if (p_dir / "val").exists() else 0
            other_dirs = [d.name for d in p_dir.iterdir() if d.is_dir() and d.name not in ("train", "val")]
            total = train_n + val_n
            expected = EXPECTED_PATCH_COUNTS.get(mpp_id, {}).get(patient, -1)
            count_match = (total == expected) if expected > 0 else True

            report_rows.append({
                "audit_type": "partner_cache",
                "mpp_id": mpp_id, "patient": patient,
                "path": str(p_dir), "exists": True,
                "train_pt": train_n, "val_pt": val_n, "total_pt": total,
                "expected": expected, "count_match": count_match,
                "other_subdirs": ";".join(other_dirs),
                "status": "OK" if count_match else "WARN",
            })
            print(f"  [{'OK' if count_match else 'WARN'}] MPP{mpp_id}_UNI/{patient}: "
                  f"train={train_n}, val={val_n}, total={total} (期望{expected})")


def audit_flat_cache(flat_root: Path, mpp_ids: list, report_rows: list):
    """审计扁平缓存 {N}/{patient}/*.pt（MPP-3 我方提取、MPP-2/XZY 外部测试）。"""
    banner(f"3b. 扁平缓存审计 (mpp_uni2h_cache/{{N}}/) — P0-3")
    print(f"  flat_cache_root: {flat_root}")

    for mpp_id in mpp_ids:
        mpp_dir = flat_root / str(mpp_id)
        if not mpp_dir.exists():
            print(f"  [MISSING] {mpp_dir}")
            report_rows.append({
                "audit_type": "flat_cache",
                "mpp_id": mpp_id, "patient": "ALL",
                "path": str(mpp_dir), "exists": False, "status": "MISSING",
            })
            continue

        for patient in TRAIN_PATIENTS + [EXTERNAL_PATIENT]:
            p_dir = mpp_dir / patient
            if not p_dir.exists():
                report_rows.append({
                    "audit_type": "flat_cache",
                    "mpp_id": mpp_id, "patient": patient,
                    "path": str(p_dir), "exists": False, "status": "MISSING",
                })
                print(f"  [MISSING] {mpp_id}/{patient}")
                continue

            n_pt = len(list(p_dir.glob("*.pt")))
            expected = EXPECTED_PATCH_COUNTS.get(mpp_id, {}).get(patient, -1)
            count_match = (n_pt == expected) if expected > 0 else True

            report_rows.append({
                "audit_type": "flat_cache",
                "mpp_id": mpp_id, "patient": patient,
                "path": str(p_dir), "exists": True,
                "n_pt": n_pt, "expected": expected, "count_match": count_match,
                "status": "OK" if count_match else "WARN",
            })
            print(f"  [{'OK' if count_match else 'WARN'}] {mpp_id}/{patient}: "
                  f"n_pt={n_pt} (期望{expected})")


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MPP1-5 统一标准重跑前置审计 (只读, 无 GPU)")
    parser.add_argument("--mpp-root", default=DEFAULT_MPP_ROOT,
                        help=f"MPP 数据根 (默认: {DEFAULT_MPP_ROOT})")
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT,
                        help=f"partner 缓存根 (MPP{{N}}_UNI/, 默认: {DEFAULT_CACHE_ROOT})")
    parser.add_argument("--flat-cache-root", default=DEFAULT_FLAT_CACHE_ROOT,
                        help=f"扁平缓存根 ({{N}}/{{patient}}/, 默认: {DEFAULT_FLAT_CACHE_ROOT})")
    parser.add_argument("--skip-cache", action="store_true",
                        help="跳过特征缓存审计（本地无缓存副本时用）")
    parser.add_argument("--output-dir", default=".",
                        help="输出目录 (默认当前目录)")
    args = parser.parse_args()

    mpp_root = Path(args.mpp_root)
    cache_root = Path(args.cache_root)
    flat_root = Path(args.flat_cache_root)
    out_dir = Path(args.output_dir)

    print(f"MPP root:         {mpp_root}")
    print(f"partner cache:    {cache_root}")
    print(f"flat cache:       {flat_root}")
    print(f"skip_cache:       {args.skip_cache}")

    report_rows = []
    candidate_rows = []

    # 1. raw 标签审计
    labels_ok = audit_raw_labels(mpp_root, report_rows)

    # 2. patch 坐标 + 步长审计
    audit_patch_coordinates(mpp_root, report_rows, candidate_rows)

    # 3. 缓存审计
    if not args.skip_cache:
        audit_partner_cache(cache_root, [1, 4], report_rows)  # MPP1_UNI, MPP4_UNI
        audit_flat_cache(flat_root, [2, 3], report_rows)       # mpp_uni2h_cache/3, /2/XZY

    # ── 写报告 ──
    report_path = out_dir / "audit_mpp_coordinates_report.csv"
    if report_rows:
        # 统一列：取所有 row key 并集
        all_keys = []
        for r in report_rows:
            for k in r.keys():
                if k not in all_keys:
                    all_keys.append(k)
        with open(report_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            for r in report_rows:
                w.writerow({k: r.get(k, "") for k in all_keys})
        print(f"\n审计报告: {report_path}")

    candidate_path = out_dir / "split_candidate_input.csv"
    if candidate_rows:
        df = pd.DataFrame(candidate_rows)
        df.to_csv(candidate_path, index=False, encoding="utf-8-sig")
        print(f"block_size 选择输入: {candidate_path}")

    banner("审计完成")
    print(f"  - raw 标签: {'OK' if labels_ok else '有 WARN, 见上方'}")
    print(f"  - 完整报告: {report_path}")
    print(f"  - block_size 输入: {candidate_path}")
    print(f"\n下一步: 把上述两个 CSV 回传本地, 据此选 block_size 并运行 generate_standard_splits.py")


if __name__ == "__main__":
    main()