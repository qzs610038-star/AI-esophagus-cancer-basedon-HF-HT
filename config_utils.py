"""
config_utils.py - PFMval 项目统一配置工具

提供配置文件加载、路径解析、设备选择等实用函数。
迁移到服务器时，只需修改 config.yaml，无需改动代码。
"""

import os
import yaml
import torch


# ============================================================
# 配置加载
# ============================================================

def get_project_root():
    """
    获取项目根目录。

    从当前文件（config_utils.py）所在目录向上搜索，
    直到找到包含 config.yaml 的目录为止。
    支持从子目录（如 histogene/, uni2h_new/ 等）调用。

    Returns:
        str: 项目根目录的绝对路径。

    Raises:
        FileNotFoundError: 如果向上搜索到文件系统根目录仍未找到 config.yaml。
    """
    # 从本文件的位置开始向上查找
    current = os.path.dirname(os.path.abspath(__file__))

    while True:
        candidate = os.path.join(current, "config.yaml")
        if os.path.isfile(candidate):
            return current

        parent = os.path.dirname(current)
        if parent == current:
            # 已到达文件系统根目录，仍未找到
            raise FileNotFoundError(
                "未找到 config.yaml。请确保项目根目录中存在该文件。\n"
                "搜索起点: " + os.path.dirname(os.path.abspath(__file__))
            )
        current = parent


def load_config(config_path=None):
    """
    加载项目配置文件（config.yaml），返回 dict。

    查找顺序:
      1. 显式传入的 config_path 参数
      2. 环境变量 PFMVAL_CONFIG 指定的路径
      3. 项目根目录下的 config.yaml（通过 get_project_root() 向上搜索）

    Args:
        config_path (str, optional): 配置文件的绝对或相对路径。默认为 None。

    Returns:
        dict: 解析后的配置字典。

    Raises:
        FileNotFoundError: 如果按顺序均未找到配置文件。
        yaml.YAMLError: 如果配置文件格式错误。
    """
    # 优先级 1：显式传入路径
    if config_path is not None:
        resolved = os.path.abspath(config_path)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"指定的配置文件不存在: {resolved}")
        return _read_yaml(resolved)

    # 优先级 2：环境变量
    env_path = os.environ.get("PFMVAL_CONFIG", "").strip()
    if env_path:
        resolved = os.path.abspath(env_path)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(
                f"环境变量 PFMVAL_CONFIG 指定的配置文件不存在: {resolved}"
            )
        return _read_yaml(resolved)

    # 优先级 3：自动搜索项目根目录
    project_root = get_project_root()
    default_path = os.path.join(project_root, "config.yaml")
    return _read_yaml(default_path)


def _read_yaml(filepath):
    """
    读取并解析 YAML 文件（内部辅助函数）。

    Args:
        filepath (str): YAML 文件的绝对路径。

    Returns:
        dict: 解析后的配置字典。
    """
    with open(filepath, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # safe_load 在文件为空时返回 None
    config = config if config is not None else {}
    path_ids = config.get("path_ids", {}) or {}
    if path_ids:
        from path_registry import get_registered_path

        paths = config.get("paths") or {}
        config["paths"] = paths
        for config_key, path_id in path_ids.items():
            paths[config_key] = str(get_registered_path(str(path_id)))
    return config


# ============================================================
# 路径解析
# ============================================================

def resolve_path(path_str, project_root=None):
    """
    解析路径字符串为绝对路径。

    - 如果 path_str 是绝对路径，直接返回（规范化后）。
    - 如果是相对路径（如 "./HYZ15040/train_patches"），
      则基于 project_root 解析。
    - project_root 为 None 时自动调用 get_project_root()。

    Args:
        path_str (str): 原始路径字符串，支持 "./" 前缀的相对路径。
        project_root (str, optional): 项目根目录绝对路径。默认 None（自动获取）。

    Returns:
        str: 规范化后的绝对路径（Windows/Linux 均兼容）。
    """
    if project_root is None:
        project_root = get_project_root()

    # os.path.isabs 在 Windows 和 Linux 均正确判断绝对路径
    if os.path.isabs(path_str):
        return os.path.normpath(path_str)

    # 相对路径：基于 project_root 拼接，再规范化
    return os.path.normpath(os.path.join(project_root, path_str))


# ============================================================
# 具体配置读取
# ============================================================

def get_data_paths(config=None):
    """
    获取数据路径配置，所有路径均解析为绝对路径。

    Args:
        config (dict, optional): 已加载的配置字典。为 None 时自动调用 load_config()。

    Returns:
        dict: 包含以下键的字典（值均为绝对路径字符串）：
            - "train_patches_dir": 训练集 patch 图像目录
            - "val_patches_dir":   验证集 patch 图像目录
            - "labels_csv_raw":    原始标签 CSV 文件路径
            - "labels_csv_zscore": Z-score 标准化后的标签 CSV 文件路径
    """
    if config is None:
        config = load_config()

    data_cfg = config.get("data", {})
    project_root = get_project_root()

    return {
        "train_patches_dir": resolve_path(
            data_cfg.get("train_patches_dir", "./HYZ15040/train_patches"),
            project_root,
        ),
        "val_patches_dir": resolve_path(
            data_cfg.get("val_patches_dir", "./HYZ15040/val_patches"),
            project_root,
        ),
        "labels_csv_raw": resolve_path(
            data_cfg.get("labels_csv_raw", "./HYZ15040_ssGSEA_scores.csv"),
            project_root,
        ),
        "labels_csv_zscore": resolve_path(
            data_cfg.get("labels_csv_zscore", "./HYZ15040_ssGSEA_scores_zscore.csv"),
            project_root,
        ),
    }


def get_hf_config(config=None):
    """
    获取 HuggingFace 相关配置。

    Token 读取优先级：
      1. config.yaml 中的 huggingface.token（非空字符串）
      2. 环境变量 HF_TOKEN

    Args:
        config (dict, optional): 已加载的配置字典。为 None 时自动调用 load_config()。

    Returns:
        dict: 包含以下键的字典：
            - "token" (str | None): HuggingFace 访问令牌，未配置时为 None。
            - "local_only" (bool):  是否强制使用本地缓存，无网络环境设为 True。
    """
    if config is None:
        config = load_config()

    hf_cfg = config.get("huggingface", {})

    # 读取 token：配置文件优先，其次环境变量
    token = hf_cfg.get("token", "").strip() or None
    if token is None:
        token = os.environ.get("HF_TOKEN", "").strip() or None

    local_only = bool(hf_cfg.get("local_only", False))

    return {
        "token": token,
        "local_only": local_only,
    }


def get_device(config=None):
    """
    获取训练设备（torch.device）。

    配置值说明：
      - "auto"（默认）：自动检测，有 CUDA 则用 GPU，否则用 CPU。
      - "cuda"：强制使用 GPU（若不可用会抛出 RuntimeError）。
      - "cpu" ：强制使用 CPU。

    Args:
        config (dict, optional): 已加载的配置字典。为 None 时自动调用 load_config()。

    Returns:
        torch.device: 解析后的设备对象。

    Raises:
        RuntimeError: 当配置为 "cuda" 但系统无可用 GPU 时。
        ValueError: 当配置了无法识别的设备字符串时。
    """
    if config is None:
        config = load_config()

    device_str = config.get("training", {}).get("device", "auto").strip().lower()

    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "配置指定了 device='cuda'，但当前系统没有可用的 CUDA GPU。\n"
                "请将 config.yaml 中的 training.device 改为 'cpu' 或 'auto'。"
            )
        return torch.device("cuda")

    if device_str == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"无法识别的设备配置: '{device_str}'。\n"
        "有效值为: 'auto', 'cuda', 'cpu'。"
    )


# ============================================================
# 路径配置扩展（服务器迁移用）
# ============================================================

# Backbone → (config_key, default_cache_dir, default_aug_dir)
_BACKBONE_CACHE_MAP = {
    'uni_cls':         ('uni_cls',         'uni2h_cache',           None),
    'uni_tokens':      ('uni_tokens',      'uni2h_cache_tokens',    'uni2h_cache_tokens_aug'),
    'uni_tokens_aug':  ('uni_tokens_aug',  'uni2h_cache_tokens',    'uni2h_cache_tokens_aug'),
    'omiclip':         ('omiclip',         'omiclip_cache',         None),
    'virchow2':        ('virchow2',        'virchow2_cache_tokens', None),
}


def get_patient_paths(patient, backbone=None, config=None):
    """
    获取单个患者的完整数据路径字典。

    路径解析优先级（每级未配置时自动 fallback 到本地默认值）：
      1. config.yaml patients.{patient}.patches_dir / labels_path（绝对路径覆盖）
      2. config.yaml paths.patch_base / ssgsea_base + 患者 subdir
      3. 项目根目录下的默认路径（与迁移前硬编码行为完全一致）

    Args:
        patient (str): 患者 ID（HYZ15040 / JFX0729 / LMZ12939）
        backbone (str or None): 特征缓存类型，None 表示不需要缓存路径
        config (dict or None): 已加载的配置，None 时自动调用 load_config()

    Returns:
        dict: {
            'train_patches': str,        # 训练集补丁目录
            'val_patches': str,          # 验证集补丁目录
            'labels_csv': str,           # 标签 CSV 文件路径
            # --- 以下仅在 backbone 非 None 时存在 ---
            'token_cache_train': str,    # backbone token 缓存训练集
            'token_cache_val': str,      # backbone token 缓存验证集
            'cache_train': str,          # 别名（兼容 omiclip 命名）
            'cache_val': str,            # 别名
            # --- 以下仅在 backbone='uni_tokens_aug' 时存在 ---
            'token_aug_train': str,      # AugMix 增强缓存训练集
            'token_aug_val': str,        # AugMix 增强缓存验证集
        }
    """
    if config is None:
        config = load_config()

    project_root = get_project_root()
    paths_cfg = config.get('paths', {}) or {}
    patients_cfg = config.get('patients', {}) or {}
    patient_cfg = patients_cfg.get(patient, {}) or {}

    # ── 补丁目录 ──
    if patient_cfg.get('patches_dir'):
        patch_root = resolve_path(patient_cfg['patches_dir'], project_root)
        train_patches = os.path.join(patch_root, 'train_patches')
        val_patches = os.path.join(patch_root, 'val_patches')
    elif paths_cfg.get('patch_base'):
        patch_base = resolve_path(paths_cfg['patch_base'], project_root)
        subdir = patient_cfg.get('patches_subdir', f'{patient}_noov_split')
        train_patches = os.path.join(patch_base, subdir, 'train_patches')
        val_patches = os.path.join(patch_base, subdir, 'val_patches')
    else:
        # 本地默认（与现有硬编码一致）
        patch_base = os.path.join(project_root, 'data_new_3ST', 'patch_noov_spilt')
        subdir = patient_cfg.get('patches_subdir', f'{patient}_noov_split')
        train_patches = os.path.join(patch_base, subdir, 'train_patches')
        val_patches = os.path.join(patch_base, subdir, 'val_patches')

    # ── 标签 CSV ──
    if patient_cfg.get('labels_path'):
        labels_csv = resolve_path(patient_cfg['labels_path'], project_root)
    elif paths_cfg.get('ssgsea_base'):
        ssgsea_base = resolve_path(paths_cfg['ssgsea_base'], project_root)
        csv_name = patient_cfg.get('labels_csv', f'{patient}_ssGSEA_zscore.csv')
        labels_csv = os.path.join(ssgsea_base, csv_name)
    else:
        ssgsea_base = os.path.join(project_root, 'data_new_3ST', 'ssGSEA_zscore')
        csv_name = patient_cfg.get('labels_csv', f'{patient}_ssGSEA_zscore.csv')
        labels_csv = os.path.join(ssgsea_base, csv_name)

    result = {
        'train_patches': train_patches,
        'val_patches': val_patches,
        'labels_csv': labels_csv,
    }

    # ── Backbone 特征缓存 ──
    if backbone:
        cache_key, default_cache_dir, default_aug_dir = _BACKBONE_CACHE_MAP.get(
            backbone, (backbone, f'{backbone}_cache', None)
        )

        caches_cfg = paths_cfg.get('caches', {}) or {}

        if caches_cfg.get(cache_key):
            cache_base = resolve_path(caches_cfg[cache_key], project_root)
        else:
            cache_base = os.path.join(project_root, default_cache_dir)

        result['token_cache_train'] = os.path.join(cache_base, patient, 'train')
        result['token_cache_val'] = os.path.join(cache_base, patient, 'val')
        result['cache_train'] = result['token_cache_train']
        result['cache_val'] = result['token_cache_val']

        # AugMix 额外缓存目录
        if backbone == 'uni_tokens_aug':
            aug_key = 'uni_tokens_aug'
            if caches_cfg.get(aug_key):
                aug_base = resolve_path(caches_cfg[aug_key], project_root)
            else:
                aug_base = os.path.join(project_root, default_aug_dir or 'uni2h_cache_tokens_aug')
            result['token_aug_train'] = os.path.join(aug_base, patient, 'train')
            result['token_aug_val'] = os.path.join(aug_base, patient, 'val')

    return result


def get_all_patient_configs(patients=None, backbone=None, config=None):
    """
    批量获取多患者路径配置。

    Args:
        patients (list or None): 患者 ID 列表，None 时默认三患者
        backbone (str or None): 传递给 get_patient_paths
        config (dict or None): 已加载的配置

    Returns:
        dict: {patient_id: path_dict}
    """
    if patients is None:
        patients = ['HYZ15040', 'JFX0729', 'LMZ12939']
    return {p: get_patient_paths(p, backbone=backbone, config=config) for p in patients}


def get_fold_config(fold, config=None):
    """
    获取三折交叉验证配置。

    Args:
        fold (int): 折号 (1/2/3)
        config (dict or None): 已加载的配置

    Returns:
        dict: {'train': [patient, ...], 'test': patient}
    """
    if config is None:
        config = load_config()

    cv_cfg = config.get('cross_validation', {}) or {}
    folds_cfg = cv_cfg.get('folds', {}) or {}

    fold_key = str(fold)
    if fold_key in folds_cfg and folds_cfg[fold_key]:
        return dict(folds_cfg[fold_key])

    # 硬编码默认值（与所有训练脚本中 FOLD_CONFIGS 一致）
    _FALLBACK_FOLDS = {
        '1': {'train': ['JFX0729', 'LMZ12939'], 'test': 'HYZ15040'},
        '2': {'train': ['HYZ15040', 'LMZ12939'], 'test': 'JFX0729'},
        '3': {'train': ['HYZ15040', 'JFX0729'], 'test': 'LMZ12939'},
    }
    return dict(_FALLBACK_FOLDS.get(fold_key, _FALLBACK_FOLDS['1']))


def get_paths_config(config=None):
    """
    获取 paths 节全部路径配置，均解析为绝对路径。

    Args:
        config (dict or None): 已加载的配置

    Returns:
        dict: paths 节各键的解析后绝对路径
    """
    if config is None:
        config = load_config()

    project_root = get_project_root()
    paths_cfg = config.get('paths', {}) or {}
    result = {}

    for key in ['patch_base', 'ssgsea_base']:
        val = paths_cfg.get(key, '')
        result[key] = resolve_path(val, project_root) if val else ''

    for section in ['caches', 'resources', 'outputs']:
        section_cfg = paths_cfg.get(section, {}) or {}
        result[section] = {}
        for key, val in section_cfg.items():
            result[section][key] = resolve_path(val, project_root) if val else ''

    return result


def get_output_dir(subdir_name=None, config=None):
    """
    获取输出目录路径。

    Args:
        subdir_name (str or None): 子目录名
        config (dict or None): 已加载的配置

    Returns:
        str: 输出目录绝对路径
    """
    if config is None:
        config = load_config()

    project_root = get_project_root()
    paths_cfg = config.get('paths', {}) or {}
    outputs_cfg = paths_cfg.get('outputs', {}) or {}

    checkpoints_root = outputs_cfg.get('checkpoints_root', '')
    if checkpoints_root:
        base = resolve_path(checkpoints_root, project_root)
    else:
        base = os.path.join(project_root, 'checkpoints')

    if subdir_name:
        return os.path.join(base, subdir_name)
    return base


def get_omiclip_checkpoint_path(config=None):
    """
    获取 OmiCLIP checkpoint 路径。

    Args:
        config (dict or None): 已加载的配置

    Returns:
        str: checkpoint 文件绝对路径
    """
    if config is None:
        config = load_config()

    project_root = get_project_root()
    paths_cfg = config.get('paths', {}) or {}
    resources_cfg = paths_cfg.get('resources', {}) or {}

    if resources_cfg.get('omiclip_checkpoint'):
        return resolve_path(resources_cfg['omiclip_checkpoint'], project_root)
    return os.path.join(project_root, 'pretrained_omiclip', 'checkpoint.pt')


def get_hf_cache_dir(config=None):
    """
    获取 HuggingFace 缓存目录。

    Args:
        config (dict or None): 已加载的配置

    Returns:
        str: HF 缓存目录绝对路径
    """
    if config is None:
        config = load_config()

    project_root = get_project_root()
    paths_cfg = config.get('paths', {}) or {}
    resources_cfg = paths_cfg.get('resources', {}) or {}

    if resources_cfg.get('hf_cache'):
        return resolve_path(resources_cfg['hf_cache'], project_root)
    return os.path.join(project_root, 'hf_cache')


def get_histogene_dir(config=None):
    """返回 histogene/ 目录的绝对路径。"""
    return os.path.join(get_project_root(), 'histogene')


def get_egnv2_dir(config=None):
    """返回 egnv2/ 目录的绝对路径。"""
    return os.path.join(get_project_root(), 'egnv2')


# ============================================================
# 快速验证入口（python config_utils.py 直接运行时执行）
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("PFMval 配置系统自检")
    print("=" * 55)

    try:
        cfg = load_config()
        print(f"[OK] 配置文件加载成功")
        print(f"     根目录: {get_project_root()}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        raise SystemExit(1)

    print("\n--- 数据路径 (data: 节，供受保护文件使用) ---")
    for key, val in get_data_paths(cfg).items():
        exists = os.path.exists(val)
        status = "存在" if exists else "不存在"
        print(f"  {key}: {val}  [{status}]")

    print("\n--- HuggingFace 配置 ---")
    hf = get_hf_config(cfg)
    token_display = (hf["token"][:6] + "...") if hf["token"] else "未配置"
    print(f"  token:      {token_display}")
    print(f"  local_only: {hf['local_only']}")

    print("\n--- 训练设备 ---")
    device = get_device(cfg)
    print(f"  device: {device}")

    print("\n--- 患者路径 (get_patient_paths) ---")
    for p in ['HYZ15040', 'JFX0729', 'LMZ12939']:
        paths = get_patient_paths(p, backbone='uni_tokens', config=cfg)
        train_ok = "存在" if os.path.exists(paths['train_patches']) else "不存在"
        labels_ok = "存在" if os.path.exists(paths['labels_csv']) else "不存在"
        cache_ok = "存在" if os.path.exists(paths.get('token_cache_train', '')) else "不存在"
        print(f"  {p}: patches[{train_ok}]  labels[{labels_ok}]  cache[{cache_ok}]")

    print("\n--- 交叉验证配置 ---")
    for f in [1, 2, 3]:
        fc = get_fold_config(f, config=cfg)
        print(f"  Fold {f}: train={fc['train']} -> test={fc['test']}")

    print("\n--- 外部资源 ---")
    omi = get_omiclip_checkpoint_path(cfg)
    print(f"  OmiCLIP checkpoint: {omi}  [{'存在' if os.path.exists(omi) else '不存在'}]")
    hf_dir = get_hf_cache_dir(cfg)
    print(f"  HF cache dir:       {hf_dir}  [{'存在' if os.path.exists(hf_dir) else '不存在'}]")

    print("\n[完成] 所有配置项检查完毕。")
