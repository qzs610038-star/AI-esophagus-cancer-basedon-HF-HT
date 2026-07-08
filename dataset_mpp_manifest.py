"""
dataset_mpp_manifest.py — split_manifest 驱动的 MPP 特征 Dataset

按 split_manifest.csv 的 split 列加载 patch 级特征, 替代旧的患者级 train/val 划分。
支持两种缓存布局自动探测:
  - partner 风格 (MPP1_UNI/MPP4_UNI): {patient}/train/*.pt + {patient}/val/*.pt
    → 按 stem 跨 train/ 和 val/ 两个子目录合并查找 (C2 约束)
  - 扁平风格 (MPP2_UNI/MPP3/mpp_uni2h_cache/3/...): {patient}/*.pt 直接存放

核心方法:
  - ManifestMPPDataset(cache_root, mpp_id, patient, split, manifest_df, label_csv, ...)
    按 split == "train" 或 "internal_val" 过滤 manifest, 加载对应 .pt + z-scored 标签
  - merge_manifest_patients(...): 多患者合并 (train 或 val split)

标签匹配: barcode (CSV 首列) == .pt stem, 1:1 精确匹配 (无行序 fallback)。

用法:
    from dataset_mpp_manifest import build_manifest_datasets
    train_ds, val_ds, ext_ds = build_manifest_datasets(
        manifest_df=manifest,
        cache_root="D:/AIPatho/qzs/pfmval_deploy_git/uni2h_cache",
        flat_cache_root="D:/AIPatho/qzs/pfmval_deploy_git/mpp_uni2h_cache",
        labels_root="mpp_standard_splits/group_1/labels",
        mpp_id=1, external_mpp_id=2, external_patient="XZY",
    )
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset

TRAIN_PATIENTS = ["HYZ15040", "JFX", "LMZ12939", "TGC", "XSL", "ZHZ"]
EXTERNAL_PATIENT = "XZY"

COORD_RE = re.compile(r'x(\d+)_y(\d+)')


def parse_xy(stem: str):
    m = COORD_RE.search(stem)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _find_pt_in_cache(cache_root: Path, mpp_id: int, patient: str, stem: str,
                     partner_style: Optional[bool] = None) -> Optional[Path]:
    """在缓存中查找 {stem}.pt, 自动探测 partner 风格或扁平风格。

    Args:
        partner_style: None=自动探测; True=只查 train/val 子目录; False=只查扁平

    缓存布局:
      partner 风格: {cache_root}/MPP{N}_UNI/{patient}/{train|val}/{stem}.pt
      扁平风格:    {cache_root}/MPP{N}_UNI/{patient}/{stem}.pt
                   或 {cache_root}/{N}/{patient}/{stem}.pt (MPP-3 旧路径)
    """
    fname = f"{stem}.pt"

    # 候选根: MPP{N}_UNI/ 优先 (partner + 扁平 MPP2_UNI)
    candidates = [
        cache_root / f"MPP{mpp_id}_UNI" / patient,
        cache_root / f"MPP{mpp_id}_UNI" / patient / "train",
        cache_root / f"MPP{mpp_id}_UNI" / patient / "val",
    ]
    # 扁平 {N}/{patient}/ (MPP-3 旧路径)
    candidates.append(cache_root / str(mpp_id) / patient)

    if partner_style is True:
        # 只查 partner 风格 (train/val 子目录)
        candidates = [cache_root / f"MPP{mpp_id}_UNI" / patient / sub
                      for sub in ("train", "val")]
    elif partner_style is False:
        # 只查扁平
        candidates = [cache_root / f"MPP{mpp_id}_UNI" / patient,
                      cache_root / str(mpp_id) / patient]

    for d in candidates:
        p = d / fname
        if p.exists():
            return p
    return None


def _detect_partner_style(cache_root: Path, mpp_id: int, patient: str) -> bool:
    """探测缓存风格: 有 train/ 或 val/ 子目录 → partner 风格; 否则扁平。"""
    p_dir = cache_root / f"MPP{mpp_id}_UNI" / patient
    if not p_dir.exists():
        # 试扁平 {N}/{patient}/
        flat = cache_root / str(mpp_id) / patient
        return False if flat.exists() else False
    return (p_dir / "train").exists() or (p_dir / "val").exists()


class ManifestMPPDataset(Dataset):
    """split_manifest 驱动: 按 split 加载 patch 级特征 + z-scored 标签。

    Args:
        manifest_df: split_manifest.csv 的 DataFrame, 含 patch_stem/patient/split/x/y
        cache_root: 特征缓存根
        mpp_id: 训练 MPP 编号
        patient: 单例患者
        split: "train" 或 "internal_val"
        labels_csv: z-scored 标签 CSV (含 barcode + 30 通路); 可为 None (外部测试用 raw/xzy)
        target_cols: 通路列名 (None 则从 labels_csv 推断)
        allow_missing: 缺失时 warn 不报错 (smoke)
        limit: 仅加载前 N 个 (本地验证加速)
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        cache_root: str,
        mpp_id: int,
        patient: str,
        split: str,
        labels_csv: str,
        target_cols: Optional[List[str]] = None,
        allow_missing: bool = False,
        limit: Optional[int] = None,
    ):
        self.cache_root = Path(cache_root)
        self.mpp_id = mpp_id
        self.patient = patient
        self.split = split

        # 过滤 manifest: 该患者该 split
        sub = manifest_df[
            (manifest_df["patient"] == patient) & (manifest_df["split"] == split)
        ].copy()
        if limit:
            sub = sub.head(limit)
        stems = sub["patch_stem"].astype(str).tolist()

        if len(stems) == 0:
            if allow_missing:
                print(f"  [WARN] {patient}/{split}: manifest 无 patch, 跳过")
                self._empty = True
                self.feat_dim = 1536
                self.target_cols = []
                self.samples = []
                return
            raise ValueError(f"{patient}/{split}: manifest 无 patch且 split={split}")

        # 探测缓存风格 (partner train/val 子目录 vs 扁平)
        partner_style = _detect_partner_style(self.cache_root, mpp_id, patient)
        # 对 partner 风格, 按 stem 查找时已自动跨 train/val 子目录合并

        # 加载标签
        label_map = {}  # stem -> np.array(30)
        if labels_csv and Path(labels_csv).exists():
            lw = pd.read_csv(labels_csv)
            first_col = lw.columns[0]  # barcode 或 patch_id
            if target_cols is None:
                skip = {first_col, "x", "y", "spot", "id", "spot_id",
                        "barcode", "index", "patch_id", "filename"}
                numeric = list(lw.select_dtypes(include=["number"]).columns)
                target_cols = [c for c in numeric if c.lower() not in skip]
            lw["_stem"] = lw[first_col].astype(str).apply(lambda x: Path(x).stem)
            for _, row in lw.iterrows():
                label_map[row["_stem"]] = row[target_cols].values.astype(np.float32)
        elif target_cols is None:
            target_cols = []

        self.target_cols = target_cols

        # 按 stem 加载 .pt 路径 + 匹配标签
        self.samples: List[Tuple[Path, torch.Tensor]] = []
        unmatched = []
        for stem in stems:
            pt_path = _find_pt_in_cache(self.cache_root, mpp_id, patient, stem,
                                       partner_style=partner_style)
            if pt_path is None:
                unmatched.append((stem, "no .pt"))
                continue
            if labels_csv and label_map:
                if stem not in label_map:
                    unmatched.append((stem, "no label"))
                    continue
                target = torch.tensor(label_map[stem], dtype=torch.float32)
            else:
                target = torch.zeros(len(target_cols), dtype=torch.float32)
            self.samples.append((pt_path, target))

        if unmatched:
            print(f"  [WARN] {patient}/{split}: {len(unmatched)}/{len(stems)} 未匹配")
            if not allow_missing:
                for stem, why in unmatched[:5]:
                    print(f"    {stem}: {why}")
                raise ValueError(f"{patient}/{split}: {len(unmatched)} 未匹配 (allow_missing=False)")

        if len(self.samples) == 0:
            if allow_missing:
                print(f"  [WARN] {patient}/{split}: 无匹配样本, 跳过")
                self._empty = True
                self.feat_dim = 1536
                return
            raise ValueError(f"{patient}/{split}: 无匹配样本")

        # 探测特征维度 + 缓存格式 (CLS [1536] 或 token [265,1536])
        feat_sample = torch.load(self.samples[0][0], map_location="cpu")
        if feat_sample.ndim == 2:
            self.feat_dim = feat_sample.shape[1]
            self._is_token_cache = True
        else:
            self.feat_dim = feat_sample.shape[0]
            self._is_token_cache = False
        self._empty = False

        print(f"  [DS] {patient}/{split}: {len(self.samples)} 样本, "
              f"feat_dim={self.feat_dim}, "
              f"partner_style={partner_style}, targets={len(self.target_cols)}")

    def __len__(self):
        if getattr(self, "_empty", False):
            return 0
        return len(self.samples)

    def __getitem__(self, idx):
        if getattr(self, "_empty", False):
            raise IndexError("Empty dataset")
        pt_path, target = self.samples[idx]
        feature = torch.load(pt_path, map_location="cpu")
        if feature.ndim == 2:
            feature = feature[0, :]  # CLS token [1536] from [265, 1536]
        return feature, target


def merge_manifest_patients(
    manifest_df: pd.DataFrame,
    cache_root: str,
    mpp_id: int,
    patients: List[str],
    split: str,
    labels_root: str,
    target_cols: Optional[List[str]] = None,
    allow_missing: bool = False,
) -> ConcatDataset:
    """合并多患者同一 split 的 Dataset (如 6 患者 train 合并, 6 患者 val 合并)。

    标签路径约定 (与 rebuild_zscore_from_manifest.py 输出一致):
      train split: {labels_root}/train/{patient}/{patient}_ssGSEA_zscore.csv
      val split:   {labels_root}/val/{patient}/{patient}_ssGSEA_zscore.csv
    """
    labels_root = Path(labels_root)
    sub_dir = "train" if split == "train" else "val"
    datasets = []
    for patient in patients:
        labels_csv = labels_root / sub_dir / patient / f"{patient}_ssGSEA_zscore.csv"
        ds = ManifestMPPDataset(
            manifest_df=manifest_df,
            cache_root=cache_root,
            mpp_id=mpp_id,
            patient=patient,
            split=split,
            labels_csv=str(labels_csv),
            target_cols=target_cols,
            allow_missing=allow_missing,
        )
        if len(ds) > 0:
            datasets.append(ds)

    if not datasets:
        raise RuntimeError(f"无可用数据集 ({split}, 6 患者均空)")
    return ConcatDataset(datasets)


def build_manifest_external(flat_cache_root: str, external_mpp_id: int,
                            external_patient: str,
                            labels_csv: str,
                            target_cols: Optional[List[str]] = None,
                            allow_missing: bool = False) -> "ManifestMPPDataset":
    """构建外部测试集 Dataset (固定 MPP-2/XZY, 复用扁平方缓存)。

    外部缓存路径 (方案 §2.2):
      {flat_cache_root}/2/XZY/*.pt  (复用现有 mpp_uni2h_cache)
      或 {flat_cache_root}/MPP2_UNI/XZY/*.pt (若已重提取到 MPP2_UNI)
    """
    flat = Path(flat_cache_root)
    # 探测两个候选位置
    candidates_xzy = [
        flat / str(external_mpp_id) / external_patient,       #mpp_uni2h_cache/2/XZY
        flat / f"MPP{external_mpp_id}_UNI" / external_patient,  # uni2h_cache/MPP2_UNI/XZY
    ]
    xzy_cache = next((d for d in candidates_xzy if d.exists()), None)
    if xzy_cache is None:
        if allow_missing:
            print(f"  [WARN] external {external_patient} 缓存不存在, 跳过")
            # 返回空 Dataset
            empty = ManifestMPPDataset.__new__(ManifestMPPDataset)
            empty._empty = True
            empty.feat_dim = 1536
            empty.target_cols = target_cols or []
            empty.samples = []
            return empty
        raise FileNotFoundError(
            f"外部测试缓存不存在: {candidates_xzy}")

    # 外部测试无 manifest split, 直接列 .pt
    pt_files = sorted(xzy_cache.glob("*.pt"))
    if not pt_files:
        raise ValueError(f"外部缓存无 .pt: {xzy_cache}")

    # 加载标签
    lw = pd.read_csv(labels_csv)
    first_col = lw.columns[0]
    if target_cols is None:
        skip = {first_col, "x", "y", "spot", "id", "spot_id",
                "barcode", "index", "patch_id", "filename"}
        numeric = list(lw.select_dtypes(include=["number"]).columns)
        target_cols = [c for c in numeric if c.lower() not in skip]

    lw["_stem"] = lw[first_col].astype(str).apply(lambda x: Path(x).stem)
    label_map = {row["_stem"]: row[target_cols].values.astype(np.float32)
                 for _, row in lw.iterrows()}

    samples = []
    unmatched = 0
    for pt in pt_files:
        if pt.stem in label_map:
            target = torch.tensor(label_map[pt.stem], dtype=torch.float32)
            samples.append((pt, target))
        else:
            unmatched += 1
    if unmatched:
        print(f"  [WARN] external {external_patient}: {unmatched}/{len(pt_files)} 无标签匹配")
    if not samples:
        raise ValueError(f"external {external_patient}: 无匹配样本")

    # 探测 feat_dim
    feat_sample = torch.load(samples[0][0], map_location="cpu")
    feat_dim = feat_sample.shape[1] if feat_sample.ndim == 2 else feat_sample.shape[0]

    # 构造 Dataset 实例 (绕过 __init__, 手动赋值)
    ds = ManifestMPPDataset.__new__(ManifestMPPDataset)
    ds.cache_root = xzy_cache.parent
    ds.mpp_id = external_mpp_id
    ds.patient = external_patient
    ds.split = "external"
    ds.samples = samples
    ds.target_cols = target_cols
    ds.feat_dim = feat_dim
    ds._is_token_cache = feat_sample.ndim == 2
    ds._empty = False
    print(f"  [DS] external {external_patient}: {len(samples)} 样本, "
          f"feat_dim={feat_dim}, targets={len(target_cols)}, cache={xzy_cache}")
    return ds


def build_manifest_datasets(
    manifest_df: pd.DataFrame,
    cache_root: str,
    flat_cache_root: str,
    labels_root: str,
    train_mpp_id: int,
    external_mpp_id: int,
    external_patient: str = "XZY",
    train_patients: Optional[List[str]] = None,
    allow_missing: bool = False,
) -> Tuple[ConcatDataset, ConcatDataset, "ManifestMPPDataset"]:
    """一行构建 train + val + external 三个数据集。

    Args:
        manifest_df: split_manifest.csv (含 patch_stem/patient/split)
        cache_root: {N}/{patient}/ 风格缓存根 (MPP2_UNI/MPP4_UNI + 新建 MPP2_UNI/MPP5_UNI)
        flat_cache_root: 扁平缓存根 (mpp_uni2h_cache, 含 /2/XZY 和 /3/{patient})
        labels_root: {splits_root}/group_{N}/labels (z-scored 标签根)
    """
    train_patients = train_patients or TRAIN_PATIENTS

    train_ds = merge_manifest_patients(
        manifest_df, cache_root, train_mpp_id, train_patients,
        split="train", labels_root=labels_root, allow_missing=allow_missing,
    )
    val_ds = merge_manifest_patients(
        manifest_df, cache_root, train_mpp_id, train_patients,
        split="internal_val", labels_root=labels_root, allow_missing=allow_missing,
    )

    # external XZY 标签: {labels_root}/external/XZY/XZY_ssGSEA_zscore_by_group_{N}_train.csv
    ext_label = (Path(labels_root) / "external" / external_patient
                 / f"{external_patient}_ssGSEA_zscore_by_group_{train_mpp_id}_train.csv")
    ext_ds = build_manifest_external(
        flat_cache_root=flat_cache_root,
        external_mpp_id=external_mpp_id,
        external_patient=external_patient,
        labels_csv=str(ext_label),
        allow_missing=allow_missing,
    )
    return train_ds, val_ds, ext_ds