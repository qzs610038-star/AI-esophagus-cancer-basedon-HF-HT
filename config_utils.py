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
    return config if config is not None else {}


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

    print("\n--- 数据路径 ---")
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

    print("\n[完成] 所有配置项检查完毕。")
