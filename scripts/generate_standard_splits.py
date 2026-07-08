#!/usr/bin/env python3
"""
scripts/generate_standard_splits.py — MPP1-5 统一标准 split 生成

按方案 §三 + §十一.2 生成每例患者各留 10% patch 的内部验证集划分。

核心算法:
  - block split: block_id = patient:floor(x/bs):floor(y/bs)，按 block 抽样
  - MPP-1/2/4 (100% 步长): 仅 block split，无 embargo
  - MPP-3/5 (50% 步长): block split + bbox overlap embargo
    (对每个 val patch, 找 bbox 重叠的 train patch 标为 embargo,
     不参与训练/验证/z-score 拟合)
  - 三档 block_size: 672 / 1120 / 1568 px，输出 split_candidate_summary.csv 供人工选择

验收标准 (方案 §三 / §七):
  - 每例患者 internal val 比例落在 8%-12%
  - train/internal val 患者集合完全一致，均覆盖六例患者
  - 同一 patch_stem 不得重复出现在不同 split
  - MPP-3/5: 不存在 iou > 0 的 train/internal_val 配对 (泄漏对数 = 0)

输入:
  - raw ssGSEA CSV: {mpp_root}/{N}/{patient}/{patient}_ssGSEA.csv
    (首列 barcode = patch_xXXX_yXXX, 后 30 通路列)
  - 或审计脚本输出的 split_candidate_input.csv (可选用其坐标范围/block_size 提示)

输出 (out_dir/group_{N}/):
  - split_manifest.csv          最终采用 block_size 的逐 patch split
  - overlap_embargo_audit.csv   MPP-3/5 的重叠审计 (MPP-1/2/4 不生成)
  - split_candidate_summary.csv  三档 block_size 候选汇总
  - split_info.json             最终采用参数 (block_size, seed, val_ratio, embargo 数)
  - split_info.csv              患者级 split 摘要 (兼容现有 train_mpp_uni2h_mlp.py 的 auto_split)

用法 (服务器):
    cd D:\\AIPatho\\qzs\\pfmval_deploy_git
    "C:\\Users\\AIPatho1\\pfmval_env\\Scripts\\python.exe" scripts/generate_standard_splits.py \\
        --mpp-root D:\\AIPatho\\Patch\\visiumhd_patch \\
        --output-root mpp_standard_splits

    # 指定最终采用 block_size (默认由候选汇总自动选, 也可强制)
    "C:\\Users\\AIPatho1\\pfmval_env\\Scripts\\python.exe" scripts/generate_standard_splits.py \\
        --final-block-size 1120

本地验证 (用 partner z-scored CSV 模拟):
    "C:\\Program Files\\Python313\\python.exe" scripts/generate_standard_splits.py \\
        --mpp-root parter_ljk_MPP1&4_patch_split_zscore \\
        --mpp-id-override 1 --use-partner-labels \\
        --output-root tmp/test_splits --final-block-size 672
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── 常量 ──
DEFAULT_MPP_ROOT = r"D:\AIPatho\Patch\visiumhd_patch"
DEFAULT_OUTPUT_ROOT = "mpp_standard_splits"

# 方案 §二 六例非 XZY 训练患者 + 外部测试
TRAIN_PATIENTS = ["HYZ15040", "JFX", "LMZ12939", "TGC", "XSL", "ZHZ"]
EXTERNAL_PATIENT = "XZY"
MPP_IDS = [1, 2, 3, 4, 5]

# 方案 §十一.2 三档 block_size 候选
BLOCK_SIZE_CANDIDATES = [672, 1120, 1568]  # 3*224, 5*224, 7*224

# 50% 步长 MPP (需 embargo)；100% 步长 MPP (无 embargo)
OVERLAP_MPPS = {3, 5}

# Patch 视野大小 (px)，用于 bbox overlap 判定
PATCH_SIZE = 224

# 验收阈值
VAL_RATIO_TARGET = 0.10
VAL_RATIO_MIN = 0.08
VAL_RATIO_MAX = 0.12
EMBARGO_RATIO_WARN = 0.30  # 方案 §十一.2: embargo/total 原则上 < 30%

# 默认 seed (方案 §五 seed=42)
DEFAULT_SEED = 42

COORD_RE = re.compile(r'x(\d+)_y(\d+)')


def parse_xy(stem: str) -> Tuple[Optional[int], Optional[int]]:
    """从 'patch_x10192_y10192' 解析 (x, y)。"""
    m = COORD_RE.search(stem)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def block_id_of(patient: str, x: int, y: int, block_size: int) -> str:
    """生成 block_id: patient:bx:by。"""
    bx = x // block_size
    by = y // block_size
    return f"{patient}:{bx}:{by}"


def load_patches_from_label_csv(label_csv: Path) -> List[Tuple[str, int, int]]:
    """从标签 CSV 读 barcode 列，解析 (stem, x, y) 列表。

    标签 CSV 首列名可能是 barcode 或 patch_id；值形如 patch_xXXX_yXXX。
    """
    df = pd.read_csv(label_csv, usecols=[0])
    first_col = df.columns[0]
    stems = df[first_col].astype(str).tolist()
    patches = []
    for s in stems:
        x, y = parse_xy(s)
        if x is not None:
            patches.append((s, x, y))
    return patches


def load_patches_from_patch_dir(patch_dir: Path) -> List[Tuple[str, int, int]]:
    """从 patch_images 目录读 .png 文件名，解析 (stem, x, y) 列表。"""
    patches = []
    for p in sorted(patch_dir.iterdir()):
        if p.suffix.lower() == ".png":
            x, y = parse_xy(p.stem)
            if x is not None:
                patches.append((p.stem, x, y))
    return patches


def load_patches_from_partner(patient: str, mpp_root: Path, mpp_id: int) -> List[Tuple[str, int, int]]:
    """从 partner z-scored 标签加载 barcode (跨 train/ 和 val/ 子目录合并)。

    partner 布局: group_{N}/train/{patient}/{patient}_ssGSEA_zscore.csv
                 group_{N}/val/{patient}/{patient}_ssGSEA_zscore.csv
    partner 的 train/val 子目录绑定的是队友的患者级划分 (如 HYZ=val, 其余=train);
    新方案每例患者都要做 patch 级 block split, 所以每例患者的 barcode 可能
    分散在 train/ 和 val/ 两个子目录, 需同时读取合并。
    """
    group_dir = mpp_root / f"group_{mpp_id}"
    patches = []
    for sub in ("train", "val"):
        csv_path = group_dir / sub / patient / f"{patient}_ssGSEA_zscore.csv"
        if csv_path.exists():
            patches.extend(load_patches_from_label_csv(csv_path))
    return patches


def assign_blocks(patches: List[Tuple[str, int, int]],
                  patient: str, block_size: int) -> Dict[str, List[Tuple[str, int, int]]]:
    """把 patches 按 block_size 分组，返回 {block_id: [(stem, x, y), ...]}。"""
    blocks = defaultdict(list)
    for stem, x, y in patches:
        bid = block_id_of(patient, x, y, block_size)
        blocks[bid].append((stem, x, y))
    return blocks


def sample_val_blocks(blocks: Dict[str, List[Tuple[str, int, int]]],
                     patient: str, total: int, seed: int,
                     val_ratio: float = VAL_RATIO_TARGET
                     ) -> Tuple[set, set]:
    """按 block 抽样直到该患者 val patch 数接近 val_ratio。

    Returns:
        val_block_ids: 被选为 val 的 block_id 集合
        val_stems: 对应的 patch stem 集合
    """
    target_val_n = int(round(total * val_ratio))
    block_ids = sorted(blocks.keys())  # 排序保证可复现
    rng = np.random.RandomState(seed + hash(patient) % 100000)

    # 按 block patch 数降序排序，优先选 patch 多的 block（减少边缘 block 偏置）
    block_sizes = [(bid, len(plist)) for bid, plist in blocks.items()]
    # 随机打乱后按 patch 数排序，兼顾随机性和"避免只有 1-2 patch 的边缘 block"
    shuffled = block_ids[:]
    rng.shuffle(shuffled)
    # 过滤掉 patch 数过少的 block（< 3），避免选到边缘碎片
    candidates = [bid for bid in shuffled if len(blocks[bid]) >= 3]
    if not candidates:
        candidates = shuffled  # 退化：所有 block 都太小

    val_block_ids = set()
    val_stems = set()
    val_n = 0
    for bid in candidates:
        if val_n >= target_val_n:
            break
        val_block_ids.add(bid)
        for stem, x, y in blocks[bid]:
            val_stems.add(stem)
            val_n += 1
    return val_block_ids, val_stems


def compute_embargo(patches: List[Tuple[str, int, int]],
                     val_stems: set,
                     patch_size: int = PATCH_SIZE
                     ) -> Tuple[set, List[dict]]:
    """对每个 val patch, 找 bbox 重叠的 train patch 标为 embargo。

    bbox = [x, y, x+patch_size, y+patch_size]。
    两个 patch bbox 重叠 iff:
        abs(x1-x2) < patch_size and abs(y1-y2) < patch_size

    Returns:
        embargo_stems: 被标 embargo 的 train patch stem 集合
        audit_rows: overlap_embargo_audit 每行 (val_patch, neighbor, dx, dy, ...)
    """
    val_set = set(val_stems)
    val_patches = [(s, x, y) for s, x, y in patches if s in val_set]
    non_val_patches = [(s, x, y) for s, x, y in patches if s not in val_set]

    embargo_stems = set()
    audit_rows = []

    # 暴力 O(V * N) — 患者级 patch 数量级 ~K，可接受
    # 如需加速可建空间索引，但此规模不必要
    for v_stem, vx, vy in val_patches:
        for n_stem, nx, ny in non_val_patches:
            dx = abs(vx - nx)
            dy = abs(vy - ny)
            if dx < patch_size and dy < patch_size:
                # bbox 重叠
                embargo_stems.add(n_stem)
                # 计算 IoU
                inter_x = max(0, patch_size - dx)
                inter_y = max(0, patch_size - dy)
                inter = inter_x * inter_y
                union = 2 * patch_size * patch_size - inter
                iou = inter / union if union > 0 else 0
                audit_rows.append({
                    "val_patch": v_stem,
                    "neighbor_patch": n_stem,
                    "dx": dx,
                    "dy": dy,
                    "iou": round(iou, 4),
                    "neighbor_final_split": "embargo",
                })
    return embargo_stems, audit_rows


def build_split_for_patient(patient: str, mpp_id: int, block_size: int,
                            patches: List[Tuple[str, int, int]],
                            seed: int, embargo: bool
                            ) -> Tuple[List[dict], dict, List[dict]]:
    """为单例患者生成 split。

    Returns:
        manifest_rows: 逐 patch split 行 (mpp_id, patient, patch_stem, x, y, split, block_id)
        summary: 该患者该 block_size 的汇总 dict
        embargo_audit_rows: overlap embargo 审计行 (embargo=True 时)
    """
    total = len(patches)
    blocks = assign_blocks(patches, patient, block_size)
    val_block_ids, val_stems = sample_val_blocks(blocks, patient, total, seed)

    val_set = set(val_stems)
    embargo_stems = set()
    embargo_audit_rows = []

    if embargo:
        embargo_stems, embargo_audit_rows = compute_embargo(patches, val_set, PATCH_SIZE)

    # 组装 manifest rows
    manifest_rows = []
    train_n = val_n = embargo_n = 0
    for stem, x, y in patches:
        bid = block_id_of(patient, x, y, block_size)
        if stem in val_set:
            split = "internal_val"
            val_n += 1
        elif stem in embargo_stems:
            split = "embargo"
            embargo_n += 1
        else:
            split = "train"
            train_n += 1
        manifest_rows.append({
            "mpp_id": mpp_id,
            "patient": patient,
            "patch_stem": stem,
            "x": x,
            "y": y,
            "split": split,
            "block_id": bid,
        })

    val_ratio = val_n / total if total > 0 else 0
    embargo_ratio = embargo_n / total if total > 0 else 0
    train_ratio = train_n / total if total > 0 else 0

    summary = {
        "mpp_id": mpp_id,
        "patient": patient,
        "block_size": block_size,
        "total_n": total,
        "train_n": train_n,
        "val_n": val_n,
        "embargo_n": embargo_n,
        "val_ratio": round(val_ratio, 4),
        "embargo_ratio": round(embargo_ratio, 4),
        "train_ratio": round(train_ratio, 4),
        "n_val_blocks": len(val_block_ids),
        "embargo_enabled": embargo,
    }
    return manifest_rows, summary, embargo_audit_rows


def verify_leakage(manifest_rows: List[dict], embargo: bool) -> Tuple[int, List[dict]]:
    """验证无 iou>0 的 train/internal_val 配对 (方案 §三.2 通过条件)。

    泄漏检查必须**患者内**: 跨患者坐标属于不同物理切片, 不算泄漏。
    对每个患者, 检查该患者 train patch 与 internal_val patch 的 bbox 重叠。
    MPP-1/2/4 (100% 步长) 理论无泄漏; MPP-3/5 (50% 步长) 应靠 embargo 清零。

    Returns:
        leakage_pairs: 泄漏对数 (患者内 train patch 与 val patch bbox 重叠)
        leakage_rows: 泄漏详情
    """
    # 按患者分组
    by_patient = defaultdict(list)
    for r in manifest_rows:
        by_patient[r["patient"]].append(r)

    leakage_pairs = 0
    leakage_rows = []
    for patient, rows in by_patient.items():
        val_patches = [(r["patch_stem"], r["x"], r["y"]) for r in rows if r["split"] == "internal_val"]
        train_patches = [(r["patch_stem"], r["x"], r["y"]) for r in rows if r["split"] == "train"]
        for v_stem, vx, vy in val_patches:
            for t_stem, tx, ty in train_patches:
                dx = abs(vx - tx)
                dy = abs(vy - ty)
                if dx < PATCH_SIZE and dy < PATCH_SIZE:
                    leakage_pairs += 1
                    leakage_rows.append({
                        "patient": patient,
                        "val_patch": v_stem,
                        "train_patch": t_stem,
                        "dx": dx, "dy": dy,
                    })
    return leakage_pairs, leakage_rows


def select_final_block_size(candidate_summary_rows: List[dict],
                            embargo: bool
                            ) -> Optional[int]:
    """按方案 §十一.2 选择准则排序三档 block_size, 选最优。

    选择准则优先级:
      1. 无泄漏 (leakage_pairs == 0)
      2. 每例患者 val_ratio 落在 8%-12%
      3. embargo_ratio 尽量低 (< 30%)
      4. val 覆盖多空间区域 (n_val_blocks 多)
    """
    if not candidate_summary_rows:
        return None
    df = pd.DataFrame(candidate_summary_rows)
    # 按 block_size 分组，聚合每 MPP 的患者级通过率
    scores = []
    for bs in BLOCK_SIZE_CANDIDATES:
        sub = df[df["block_size"] == bs]
        if sub.empty:
            continue
        # 患者级通过条件
        n_patients = sub["patient"].nunique()
        val_in_range = ((sub["val_ratio"] >= VAL_RATIO_MIN) & (sub["val_ratio"] <= VAL_RATIO_MAX)).sum()
        embargo_under_30 = (sub["embargo_ratio"] < EMBARGO_RATIO_WARN).sum() if embargo else n_patients
        mean_embargo = sub["embargo_ratio"].mean() if embargo else 0
        mean_n_val_blocks = sub["n_val_blocks"].mean()
        # 评分: 通过率主排序，embargo 低次排序
        score = (val_in_range * 100) + (embargo_under_30 * 50) - (mean_embargo * 100) + (mean_n_val_blocks * 0.1)
        scores.append((bs, score, val_in_range, embargo_under_30, mean_embargo, mean_n_val_blocks))

    scores.sort(key=lambda t: -t[1])
    print(f"  block_size 候选评分 (embargo={embargo}):")
    for bs, score, vir, eu30, me, nb in scores:
        print(f"    bs={bs:5d}: score={score:.2f}, val_in_range={vir}/{scores[0][2] if False else n_patients}, "
              f"embargo<30%={eu30}, mean_embargo={me:.3f}, mean_n_val_blocks={nb:.1f}")
    return scores[0][0]


def generate_for_mpp(mpp_id: int, mpp_root: Path, output_root: Path,
                     final_block_size: Optional[int], seed: int,
                     label_source: str) -> Tuple[bool, str]:
    """为单个 MPP 生成三档候选 + 最终 split。

    Args:
        label_source: "raw" (从 {mpp_root}/{N}/{patient}/{patient}_ssGSEA.csv 读 barcode)
                      "partner" (从 partner z-scored CSV 读, 本地验证用)
        final_block_size: 若 None, 自动选; 否则强制用指定值
    """
    embargo = mpp_id in OVERLAP_MPPS
    print(f"\n{'=' * 70}")
    print(f"  MPP-{mpp_id} (embargo={embargo}, label_source={label_source})")
    print(f"{'=' * 70}")

    # ── 加载每例患者 patches ──
    patient_patches: Dict[str, List[Tuple[str, int, int]]] = {}
    for patient in TRAIN_PATIENTS:
        if label_source == "raw":
            label_csv = mpp_root / str(mpp_id) / patient / f"{patient}_ssGSEA.csv"
            if not label_csv.exists():
                return False, f"raw label missing: {label_csv}"
            patches = load_patches_from_label_csv(label_csv)
        else:
            # partner: 跨 train/ 和 val/ 子目录合并 (HYZ15040 在 val/, 其余在 train/)
            patches = load_patches_from_partner(patient, mpp_root, mpp_id)
            if not patches:
                return False, f"partner label missing for {patient} in group_{mpp_id}"
        patient_patches[patient] = patches
        print(f"  {patient}: {len(patches)} patches")

    # ── 三档候选 split ──
    candidate_summary_rows = []
    candidate_manifests: Dict[int, List[dict]] = {}  # block_size -> manifest_rows
    candidate_embargo_audits: Dict[int, List[dict]] = {}

    for bs in BLOCK_SIZE_CANDIDATES:
        print(f"\n  block_size = {bs} px")
        all_manifest = []
        all_summary = []
        all_embargo_audit = []
        for patient in TRAIN_PATIENTS:
            patches = patient_patches[patient]
            m_rows, s, ea_rows = build_split_for_patient(
                patient, mpp_id, bs, patches, seed, embargo)
            all_manifest.extend(m_rows)
            all_summary.append(s)
            all_embargo_audit.extend(ea_rows)
            print(f"    {patient}: total={s['total_n']}, train={s['train_n']}, "
                  f"val={s['val_n']} ({s['val_ratio']*100:.1f}%), "
                  f"embargo={s['embargo_n']} ({s['embargo_ratio']*100:.1f}%), "
                  f"n_val_blocks={s['n_val_blocks']}")

        # 泄漏检查 (每档都做，embargo 档应=0)
        leakage_pairs, leakage_rows = verify_leakage(all_manifest, embargo)
        print(f"    泄漏对数 (train<->val bbox 重叠): {leakage_pairs}")

        for s in all_summary:
            s["leakage_pairs"] = leakage_pairs
            candidate_summary_rows.append(s)
        candidate_manifests[bs] = all_manifest
        candidate_embargo_audits[bs] = all_embargo_audit

    # ── 选最终 block_size ──
    if final_block_size is None:
        chosen_bs = select_final_block_size(candidate_summary_rows, embargo)
    else:
        chosen_bs = final_block_size
    print(f"\n  最终采用 block_size = {chosen_bs} px")

    # ── 验证通过条件 ──
    final_manifest = candidate_manifests[chosen_bs]
    final_embargo_audit = candidate_embargo_audits[chosen_bs]
    final_summary = [s for s in candidate_summary_rows if s["block_size"] == chosen_bs]

    # 验证: 每例患者 val_ratio 8-12%
    val_ratio_violations = [s for s in final_summary
                            if not (VAL_RATIO_MIN <= s["val_ratio"] <= VAL_RATIO_MAX)]
    # 验证: MPP-3/5 泄漏=0
    final_leakage = final_summary[0]["leakage_pairs"] if final_summary else -1

    ok = True
    if val_ratio_violations:
        ok = False
        print(f"  [WARN] {len(val_ratio_violations)} 例患者 val_ratio 超出 8%-12%:")
        for s in val_ratio_violations:
            print(f"    {s['patient']}: {s['val_ratio']*100:.1f}%")
    if embargo and final_leakage != 0:
        ok = False
        print(f"  [ERROR] MPP-{mpp_id} 泄漏未清零: {final_leakage} 对 (违反方案 §三.2 通过条件)")
    if embargo:
        embargo_high = [s for s in final_summary if s["embargo_ratio"] >= EMBARGO_RATIO_WARN]
        if embargo_high:
            print(f"  [WARN] {len(embargo_high)} 例患者 embargo_ratio >= 30% (方案建议降级该 MPP 有效样本量结论):")
            for s in embargo_high:
                print(f"    {s['patient']}: {s['embargo_ratio']*100:.1f}%")

    # ── 写文件 ──
    out_dir = output_root / f"group_{mpp_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # split_manifest.csv
    manifest_path = out_dir / "split_manifest.csv"
    pd.DataFrame(final_manifest).to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print(f"  写出: {manifest_path} ({len(final_manifest)} 行)")

    # split_candidate_summary.csv
    cand_path = out_dir / "split_candidate_summary.csv"
    pd.DataFrame(candidate_summary_rows).to_csv(cand_path, index=False, encoding="utf-8-sig")
    print(f"  写出: {cand_path}")

    # overlap_embargo_audit.csv (仅 embargo MPP)
    if embargo:
        audit_path = out_dir / "overlap_embargo_audit.csv"
        if final_embargo_audit:
            pd.DataFrame(final_embargo_audit).to_csv(audit_path, index=False, encoding="utf-8-sig")
            print(f"  写出: {audit_path} ({len(final_embargo_audit)} embargo 邻居对)")
        else:
            # 写空文件(含表头)表示无 embargo
            pd.DataFrame(columns=["val_patch", "neighbor_patch", "dx", "dy", "iou",
                                  "neighbor_final_split"]).to_csv(audit_path, index=False, encoding="utf-8-sig")
            print(f"  写出: {audit_path} (无 embargo 邻居)")

    # split_info.json (最终参数)
    info_json = {
        "mpp_id": mpp_id,
        "embargo_enabled": embargo,
        "overlap_policy": "bbox_embargo" if embargo else "none_100pct_stride",
        "block_size": chosen_bs,
        "block_size_candidates": BLOCK_SIZE_CANDIDATES,
        "seed": seed,
        "val_ratio_target": VAL_RATIO_TARGET,
        "val_ratio_range": [VAL_RATIO_MIN, VAL_RATIO_MAX],
        "fit_patients": TRAIN_PATIENTS,
        "excluded_from_zscore_fit": ["internal_val", "embargo", "external_test"],
        "patch_size_for_bbox": PATCH_SIZE,
        "leakage_pairs": final_leakage,
        "per_patient_summary": final_summary,
        "status": "OK" if ok else "WARN",
    }
    info_path = out_dir / "split_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info_json, f, indent=2, ensure_ascii=False)
    print(f"  写出: {info_path}")

    # split_info.csv (兼容现有 train_mpp_uni2h_mlp.py 的 auto_split 读取格式)
    # 现有格式 columns=[split, patient], 新方案 val 是 patch 级而非患者级,
    # 这里只写患者级摘要供 preflight 展示; 实际 patch 级 split 在 split_manifest.csv
    info_csv_rows = [{"split": "train", "patient": p} for p in TRAIN_PATIENTS]
    info_csv_rows.append({"split": "external", "patient": EXTERNAL_PATIENT})
    info_csv_path = out_dir / "split_info.csv"
    pd.DataFrame(info_csv_rows).to_csv(info_csv_path, index=False, encoding="utf-8-sig")
    print(f"  写出: {info_csv_path} (患者级摘要, patch 级见 split_manifest.csv)")

    return ok, ("OK" if ok else "WARN — 见上方 WARNING/ERROR")


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MPP1-5 统一标准 split 生成 (block + embargo)")
    parser.add_argument("--mpp-root", default=DEFAULT_MPP_ROOT,
                        help=f"MPP 数据根 (默认: {DEFAULT_MPP_ROOT})")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT,
                        help=f"输出根目录 (默认: {DEFAULT_OUTPUT_ROOT})")
    parser.add_argument("--mpp-ids", default="1,2,3,4,5",
                        help="处理的 MPP 编号, 逗号分隔 (默认 1,2,3,4,5)")
    parser.add_argument("--final-block-size", type=int, default=None,
                        choices=BLOCK_SIZE_CANDIDATES,
                        help="强制最终 block_size (默认自动选; 候选 672/1120/1568)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"随机种子 (默认 {DEFAULT_SEED})")
    parser.add_argument("--use-partner-labels", action="store_true",
                        help="本地验证: mpp-root 指向 partner z-scored 根目录")
    parser.add_argument("--mpp-id-override", type=int, default=None,
                        help="本地验证: 强制 MPP 编号 (配合 --use-partner-labels)")
    args = parser.parse_args()

    mpp_root = Path(args.mpp_root)
    output_root = Path(args.output_root)
    mpp_ids = [int(x) for x in args.mpp_ids.split(",")]

    print(f"MPP root:       {mpp_root}")
    print(f"Output root:    {output_root}")
    print(f"MPP ids:        {mpp_ids}")
    print(f"Seed:           {args.seed}")
    print(f"Final block_size: {args.final_block_size or '自动选'}")
    print(f"Label source:   {'partner' if args.use_partner_labels else 'raw'}")

    label_source = "partner" if args.use_partner_labels else "raw"

    # 若 override, 只处理这一个 MPP
    if args.mpp_id_override is not None:
        mpp_ids = [args.mpp_id_override]

    all_ok = True
    for mpp_id in mpp_ids:
        ok, msg = generate_for_mpp(mpp_id, mpp_root, output_root,
                                   args.final_block_size, args.seed, label_source)
        if not ok:
            all_ok = False
        print(f"  MPP-{mpp_id}: {msg}")

    print(f"\n{'=' * 70}")
    if all_ok:
        print(f"  全部 MPP split 生成完成 (OK)")
    else:
        print(f"  部分 MPP 有 WARN/ERROR, 请检查上方输出")
        print(f"  [MPP-3/5] 若泄漏未清零, 严禁启动正式训练 (方案 §三.2)")
    print(f"{'=' * 70}")
    print(f"\n下一步:")
    print(f"  1. 人工查看 split_candidate_summary.csv, 确认 block_size 选择合理")
    print(f"  2. 运行 scripts/rebuild_zscore_from_manifest.py 生成 train-only z-score")
    print(f"  3. MPP-3/5: 确认 overlap_embargo_audit.csv 中无 train<->val 泄漏")


if __name__ == "__main__":
    main()