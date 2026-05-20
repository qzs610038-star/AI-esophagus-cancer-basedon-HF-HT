"""
批量提取 data_new_3ST 三个数据集的 UNI2-h 特征缓存
"""
import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from uni2h.uni2h_utils import load_uni2h_backbone, extract_and_cache_features
from config_utils import get_hf_config, get_device, get_patient_paths

def main():
    device = get_device()
    print(f"设备: {device}")

    # 加载UNI2-h模型（只加载一次）
    hf_config = get_hf_config()
    print("正在加载 UNI2-h 模型...")
    backbone, transform, feat_dim = load_uni2h_backbone(
        token=hf_config.get('token'), device=device
    )
    print(f"UNI2-h 模型已加载，特征维度: {feat_dim}")

    # 从 config_utils 获取三个患者的路径
    datasets = []
    for p in ['HYZ15040', 'JFX0729', 'LMZ12939']:
        pc = get_patient_paths(p, backbone='uni_cls')
        datasets.append({
            'name': p,
            'train_patches': Path(pc['train_patches']),
            'val_patches': Path(pc['val_patches']),
            'train_cache': Path(pc['token_cache_train']),
            'val_cache': Path(pc['token_cache_val']),
        })

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"处理数据集: {ds['name']}")
        print(f"{'='*60}")
        
        for split, patches_key, cache_key in [
            ('train', 'train_patches', 'train_cache'),
            ('val', 'val_patches', 'val_cache'),
        ]:
            patches_dir = ds[patches_key]
            cache_dir = ds[cache_key]
            
            # 统计现有状态
            n_patches = len(list(patches_dir.glob('*.png'))) if patches_dir.exists() else 0
            cache_dir.mkdir(parents=True, exist_ok=True)
            n_cached = len(list(cache_dir.glob('*.pt')))
            
            print(f"\n  [{split}] patches: {n_patches}, 已缓存: {n_cached}")
            
            if n_patches == 0:
                print(f"  [{split}] 跳过 - 无补丁文件")
                continue
            
            # 提取特征（rebuild=False，跳过已存在的）
            n_new = extract_and_cache_features(
                backbone, transform,
                str(patches_dir), str(cache_dir),
                device, rebuild=False
            )
            
            n_total = len(list(cache_dir.glob('*.pt')))
            print(f"  [{split}] 新提取: {n_new}, 总缓存: {n_total}")
    
    # 释放模型
    del backbone
    torch.cuda.empty_cache()
    
    # 最终统计
    print(f"\n{'='*60}")
    print("特征提取完成！最终缓存统计：")
    print(f"{'='*60}")
    for ds in datasets:
        train_n = len(list(ds['train_cache'].glob('*.pt'))) if ds['train_cache'].exists() else 0
        val_n = len(list(ds['val_cache'].glob('*.pt'))) if ds['val_cache'].exists() else 0
        print(f"  {ds['name']}: train={train_n}, val={val_n}")

if __name__ == '__main__':
    main()
