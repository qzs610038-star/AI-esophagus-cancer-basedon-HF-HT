"""
dataset_mpp.py — MPP 实验特征缓存 + 标签 Dataset

读取预提取的 UNI2-h CLS token 特征缓存（.pt）和 z-score 标准化标签（csv），
按坐标 (x, y) 从文件名 stem 匹配标签，返回 (feature[1536], target[30])。

用法:
    ds = MPPFeatureDataset(cache_dir="mpp_uni2h_cache/3/HYZ15040",
                           labels_csv="mpp_uni2h_cache/labels/mpp3_HYZ15040_zscored.csv")
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset


def parse_coordinates(stem: str) -> Tuple[Optional[int], Optional[int]]:
    """从文件名 stem（如 'patch_x4641_y16969'）解析 (x, y)。"""
    match = re.search(r'x(\d+)_y(\d+)', stem)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


class MPPFeatureDataset(Dataset):
    """从预提取缓存读 UNI2-h CLS 特征 + 标准化标签。

    按坐标 (x, y) 匹配 .pt 文件与标签，确保稳健的 patch→label 对应。
    若 CSV 不含 x/y 坐标列，退化为按文件名 stem 匹配（假设有一 ID 列）。

    Args:
        cache_dir: mpp_uni2h_cache/{mpp_id}/{patient}/ 目录，内含 *.pt 文件
        labels_csv: 同一患者的 z-score 标准化标签 CSV
        target_cols: 目标通路列名（若 None 则用 CSV 中所有非坐标/ID 的数值列）
        allow_missing: 若 True，缺失时打印警告而非报错 (smoke test 模式)
    """

    def __init__(
        self,
        cache_dir: str,
        labels_csv: str,
        target_cols: Optional[List[str]] = None,
        allow_missing: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            if allow_missing:
                print(f"  [WARN] 特征缓存目录不存在，跳过: {self.cache_dir}")
                self._empty = True
                self.feat_dim = 1536
                return
            raise FileNotFoundError(f"特征缓存目录不存在: {self.cache_dir}")

        # 读标签 CSV
        self.labels_df = pd.read_csv(labels_csv)
        all_cols = list(self.labels_df.columns)

        # 识别坐标列
        coord_cols = [c for c in ["x", "y", "X", "Y"] if c in all_cols]

        # 重复匹配键会被 dict/set_index 静默覆盖，必须在任何匹配前硬失败。
        possible_id_cols = ["patch", "filename", "spot", "id", "patch_id",
                            "spot_id", "barcode"]
        id_col = next((c for c in possible_id_cols if c in all_cols), None)
        duplicate_checks = []
        if len(coord_cols) >= 2:
            duplicate_checks.append((coord_cols[:2], self.labels_df.duplicated(coord_cols[:2], keep=False)))
        if id_col:
            normalized_id = self.labels_df[id_col].astype(str).apply(lambda value: Path(value).stem)
            duplicate_checks.append(([id_col], normalized_id.duplicated(keep=False)))
        duplicate_key = pd.Series(False, index=self.labels_df.index)
        duplicate_key_names = []
        for names, duplicate in duplicate_checks:
            if duplicate.any():
                duplicate_key |= duplicate
                duplicate_key_names.extend(names)
        if duplicate_key.any():
            duplicate_key_names = list(dict.fromkeys(duplicate_key_names))
            examples = self.labels_df.loc[duplicate_key, duplicate_key_names].head(5).to_dict("records")
            raise ValueError(
                f"duplicate label key hard gate: labels_csv={labels_csv}, "
                f"key={duplicate_key_names}, examples={examples}"
            )

        # 识别目标通路列
        if target_cols is not None:
            self.target_cols = target_cols
        else:
            # 去掉坐标/ID 列后剩下的数值列即通路列
            skip_names = {"x", "y", "X", "Y", "spot", "id", "spot_id",
                          "barcode", "index", "patch_id", "filename"}
            numeric_cols = list(self.labels_df.select_dtypes(include=["number"]).columns)
            self.target_cols = [c for c in numeric_cols if c.lower() not in skip_names]

        n_targets = len(self.target_cols)
        print(f"  [DS] 标签 CSV: {len(self.labels_df)} 行, {n_targets} 目标通路")

        # 列出 .pt 文件
        pt_files = sorted(self.cache_dir.glob("*.pt"))
        if len(pt_files) == 0:
            if allow_missing:
                print(f"  [WARN] 缓存目录无 .pt 文件，跳过: {self.cache_dir}")
                self._empty = True
                self.feat_dim = 1536
                return
            raise ValueError(f"缓存目录无 .pt 文件: {self.cache_dir}")

        # 确定匹配策略
        self.samples: List[Tuple[str, torch.Tensor]] = []  # (stem, target_tensor)
        unmatched_stems = []

        if len(coord_cols) >= 2:
            # 策略 1: 坐标匹配（最稳健）
            x_col, y_col = coord_cols[0], coord_cols[1]
            coord_map = {}
            for idx, row in self.labels_df.iterrows():
                coord_map[(int(row[x_col]), int(row[y_col]))] = idx

            n_match = 0
            for pt in pt_files:
                x, y = parse_coordinates(pt.stem)
                if x is not None and (x, y) in coord_map:
                    row_idx = coord_map[(x, y)]
                    target = self.labels_df.iloc[row_idx][self.target_cols].values.astype(np.float32)
                    self.samples.append((pt.stem, torch.tensor(target, dtype=torch.float32)))
                    n_match += 1
                else:
                    unmatched_stems.append(pt.stem)

            print(f"  [DS] 坐标匹配: {n_match}/{len(pt_files)} 匹配, "
                  f"{len(unmatched_stems)} 未匹配")
        else:
            # 策略 2: 无坐标列 — 尝试按 ID 列匹配
            possible_id_cols = ["patch", "filename", "spot", "id", "patch_id",
                                "spot_id", "barcode"]
            id_col = next((c for c in possible_id_cols if c in all_cols), None)

            if id_col:
                self.labels_df["_id_stem"] = self.labels_df[id_col].astype(str).apply(
                    lambda x: Path(x).stem
                )
                label_map = self.labels_df.set_index("_id_stem")[self.target_cols]

                for pt in pt_files:
                    if pt.stem in label_map.index:
                        target = label_map.loc[pt.stem].values.astype(np.float32)
                        self.samples.append((pt.stem, torch.tensor(target, dtype=torch.float32)))
                    else:
                        unmatched_stems.append(pt.stem)

                print(f"  [DS] ID 匹配: {len(self.samples)}/{len(pt_files)} 匹配, "
                      f"{len(unmatched_stems)} 未匹配")
            else:
                # 策略 3: 无坐标也无 ID 列 — 按行序匹配（脆弱，仅作 fallback）
                if len(pt_files) != len(self.labels_df):
                    print(f"  [WARN] .pt 数 ({len(pt_files)}) 与标签行数 ({len(self.labels_df)}) 不一致")
                    if not allow_missing:
                        raise ValueError(
                            f".pt 数 ({len(pt_files)}) 与标签行数 ({len(self.labels_df)}) 不一致，"
                            "且 CSV 无坐标/ID 列可匹配。请检查数据或使用 --allow-missing 跳过。"
                        )

                for i, pt in enumerate(pt_files):
                    if i < len(self.labels_df):
                        target = self.labels_df.iloc[i][self.target_cols].values.astype(np.float32)
                        self.samples.append((pt.stem, torch.tensor(target, dtype=torch.float32)))
                    else:
                        unmatched_stems.append(pt.stem)

                print(f"  [DS] 行序匹配: {len(self.samples)} 样本, "
                      f"{len(unmatched_stems)} 未匹配 (fallback)")

        if unmatched_stems and not allow_missing:
            print(f"  [WARN] {len(unmatched_stems)} 个 patch 无匹配标签 (已跳过)")
            print(f"  首个未匹配: {unmatched_stems[:5]}")

        if len(self.samples) == 0:
            if allow_missing:
                print(f"  [WARN] 无匹配样本，跳过: {self.cache_dir}")
                self._empty = True
                self.feat_dim = 1536
                return
            raise ValueError(f"无匹配样本: {self.cache_dir}")

        feat_sample = torch.load(self.cache_dir / f"{self.samples[0][0]}.pt",
                                 map_location="cpu")
        # 队友缓存可能存的是全部 token [265, 1536] 而非仅 CLS [1536]
        # 检测 2D → 取 CLS (第 0 行)；1D → 直接用
        if feat_sample.ndim == 2:
            self.feat_dim = feat_sample.shape[1]  # 1536
            self._is_token_cache = True
        else:
            self.feat_dim = feat_sample.shape[0]
            self._is_token_cache = False
        self._empty = False

    def __len__(self) -> int:
        if hasattr(self, "_empty") and self._empty:
            return 0
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self, "_empty") and self._empty:
            raise IndexError("Empty dataset")
        stem = self.samples[idx][0]
        feature = torch.load(self.cache_dir / f"{stem}.pt", map_location="cpu")
        # 若缓存存的是全部 token [265, 1536]，取 CLS token [1536]
        if feature.ndim == 2:
            feature = feature[0, :]
        target = self.samples[idx][1]
        return feature, target


def merge_mpp_patients(
    cache_root: str,
    mpp_id: int,
    patients: List[str],
    labels_root: str,
    target_cols: Optional[List[str]] = None,
    allow_missing: bool = False,
    partner: bool = False,
) -> ConcatDataset:
    """合并多患者的 MPPFeatureDataset。

    Args:
        cache_root: mpp_uni2h_cache 根目录
        mpp_id: MPP 编号
        patients: 患者列表
        labels_root: 标准化标签目录
        target_cols: 目标通路列
        allow_missing: 若 True，缺失患者仅打印警告不报错 (smoke 模式)
                      若 False (默认)，缺失患者直接报错中止 (正式训练)
        partner: 若 True，使用队友目录布局 (MPP{N}_UNI/ + group_{N}/train/)
    """
    datasets = []
    for patient in patients:
        if partner:
            cache_dir = f"{cache_root}/MPP{mpp_id}_UNI/{patient}/train"
            labels_csv = f"{labels_root}/group_{mpp_id}/train/{patient}/{patient}_ssGSEA_zscore.csv"
        else:
            cache_dir = f"{cache_root}/{mpp_id}/{patient}"
            labels_csv = f"{labels_root}/mpp{mpp_id}_{patient}_zscored.csv"
        try:
            ds = MPPFeatureDataset(
                cache_dir=cache_dir,
                labels_csv=labels_csv,
                target_cols=target_cols,
                allow_missing=allow_missing,
            )
            if len(ds) > 0:
                datasets.append(ds)
            print(f"  [DS] {patient}: {len(ds)} 样本, "
                  f"feat_dim={ds.feat_dim}, targets={len(ds.target_cols)}")
        except (FileNotFoundError, ValueError) as e:
            if allow_missing:
                print(f"  [WARN] 跳过 {patient}: {e}")
            else:
                raise RuntimeError(
                    f"患者 {patient} 数据缺失 (allow_missing=False): {e}\n"
                    "使用 --allow-missing 跳过缺失患者 (仅 smoke 测试)"
                ) from e

    if len(datasets) == 0:
        raise RuntimeError("无可用数据集 (所有患者均缺失)")

    return ConcatDataset(datasets)
