"""Local-only builder. The uploaded package never reads files outside itself for code."""
import json
from pathlib import Path
import shutil
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
server = yaml.safe_load((ROOT / "configs/server_paths.yaml").read_text(encoding="utf-8"))
paths = server["paths"]
checks = []


def add(identifier, label, kind, required=True, source="configs/server_paths.yaml:paths", **kwargs):
    checks.append(dict(id=identifier, label=label, kind=kind, required=required, source=source, **kwargs))


add("runtime", "Windows 与实际 Python", "runtime", source="configs/server_paths.yaml:runtime")
for module, label, required in [("torch", "PyTorch", True), ("torchvision", "图像预处理 torchvision", True),
                                ("timm", "模型库 timm", True), ("numpy", "NumPy", True), ("pandas", "pandas", True),
                                ("PIL", "Pillow 图像读取", True), ("scipy", "SciPy", False),
                                ("sklearn", "scikit-learn", False), ("h5py", "HDF5 读取", False),
                                ("anndata", "AnnData 读取", False)]:
    add("import_" + module, label, "module", required, source="项目数据/模型读取代码的实际依赖", module=module)
add("cuda", "CUDA 小张量计算（默认 GPU 0）", "cuda", source="模型运行环境")

hf = paths["shared_huggingface_cache"]["path"]
add("uni2_weights", "UNI2-h 权重抽样读取（CPU）", "weights", path=hf + r"\hub\models--MahmoodLab--UNI2-h\snapshots",
    source="shared_huggingface_cache + uni2h/uni2h_utils.py 本地缓存布局")
for key in ["shared_cache_root", "shared_huggingface_cache", "shared_torch_cache", "mpp_data_root", "server_mpp_flat_cache", "server_mpp_partner_cache"]:
    add("path_" + key, key + " 目录", "directory", False, path=paths[key]["path"])

data_root = paths["mpp_data_root"]["path"]
flat = paths["server_mpp_flat_cache"]["path"]
partner = paths["server_mpp_partner_cache"]["path"]
for patient in ["HYZ15040", "JFX", "LMZ12939", "TGC", "XSL", "XZY", "ZHZ"]:
    add("mpp2_image_" + patient, "MPP2 " + patient + " 图像读取", "image",
        path=data_root + "\\2\\" + patient + "\\patch_images", source="templates:mpp_patient_patch_images (group=2)")
    add("mpp2_label_" + patient, "MPP2 " + patient + " 原始标签读取", "csv",
        path=data_root + "\\2\\" + patient + "\\" + patient + "_ssGSEA.csv", source="templates:mpp_patient_raw_ssgsea (group=2)")
    candidates = []
    for base in [partner, flat]:
        candidates += [base + "\\MPP2_UNI\\" + patient, base + "\\2\\" + patient]
    add("mpp2_cache_" + patient, "MPP2 " + patient + " 1536 维特征读取", "cache_candidates",
        candidates=candidates, source="server_mpp_partner_cache/server_mpp_flat_cache + dataset_mpp_manifest.py")

for identifier, label, kind in [
    ("phase2_patch_root", "旧 Phase2 图像抽样", "image"),
    ("phase2_ssgsea_zscore_root", "旧 Phase2 标准化标签抽样", "csv"),
    ("server_mpp_standard_splits", "服务器历史划分 CSV 抽样", "csv"),
    ("server_mpp2_frozen_baseline_checkpoint", "冻结基线 checkpoint 读取", "weights"),
    ("server_phase3_escc_uni2h_features", "Phase3 UNI2-h 特征抽样", "tensor"),
    ("server_phase3_escc_mpp2_raw_scores", "Phase3 MPP2 原始特征抽样", "tensor"),
]:
    add(identifier, label, kind, False, path=paths[identifier]["path"])

for identifier, label, kind in [
    ("phase2_token_cache_hyz_train", "历史记录：HYZ train token 缓存", "tensor"),
    ("phase2_token_cache_hyz_val", "历史记录：HYZ val token 缓存", "tensor"),
    ("hf_cache_openmidnight_dir", "历史记录：OpenMidnight 缓存目录", "directory"),
    ("torch_hub_checkpoints", "历史记录：Torch Hub 权重抽样", "weights"),
]:
    add(identifier, label, kind, False, source="configs/server_paths.yaml:md_import（历史快照，待服务器核验）",
        path=server["md_import"]["paths"][identifier]["path"])

config = dict(python_interpreter=server["runtime"]["python_interpreter"], runs_root=paths["server_manual_runs"]["path"],
              checks=checks, source_record=dict(file="configs/server_paths.yaml", recorded_at=server["updated_at"]))
(HERE / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
package = dict(experiment_id=HERE.name, code_version="v002", entrypoint="src/check_server.py", args=[],
               demo_only=False, diagnostic_only=True, code_sources=[{"path": "experiments/_template/run.ps1,runner.py", "date": "2026-09-05"}])
(HERE / "package.json").write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
shutil.copyfile(ROOT / "experiments/_template/runner.py", HERE / "runner.py")
launcher = (ROOT / "experiments/_template/run.ps1").read_text(encoding="utf-8")
launcher = launcher.replace("    demo_only = $package.demo_only", "    demo_only = $package.demo_only\n    diagnostic_only = $true")
launcher = launcher.replace("exit $exitCode", 'Write-Host "Copy this return folder: $runDir"\nexit $exitCode')
(HERE / "run.ps1").write_text(launcher, encoding="utf-8")
(HERE / "run.cmd").write_text('@echo off\npowershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"\nset "result=%errorlevel%"\npause\nexit /b %result%\n', encoding="ascii")
(HERE / "requirements.txt").write_text("# Do not install automatically: this package checks existing server dependencies.\n# Core: torch torchvision timm numpy pandas Pillow\n# Additional: scipy scikit-learn h5py anndata\n", encoding="utf-8")
deliverables = HERE / "deliverables"
deliverables.mkdir(exist_ok=True)
destination = deliverables / HERE.name
files = ["README.md", "config.json", "package.json", "runner.py", "run.ps1", "run.cmd", "requirements.txt", "src/check_server.py"]
for name in files:
    target = destination / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE / name, target)
print(json.dumps({"checks": len(checks), "copy_folder": str(destination), "files": len(files)}, ensure_ascii=False))
