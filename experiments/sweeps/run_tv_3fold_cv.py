"""
TV Loss 最优配置三折交叉验证
=============================
P0 sweep 找到最优 (tv_weight, tv_mode) 后，在 Fold2/Fold3 上验证一致性。

用法:
  python run_tv_3fold_cv.py --tv_weight 0.05 --tv_mode l1
  (默认使用 P0 sweep 中 PCC 最高的配置，也可手动指定)
"""
import subprocess
import sys
import re
from datetime import datetime
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent
RESULT_DIR = PROJECT_ROOT / "tv_sweep_results"
CV_RESULT_DIR = PROJECT_ROOT / "tv_3fold_cv_results"
CV_RESULT_DIR.mkdir(exist_ok=True)

PYTHON = r"C:\Program Files\Python313\python.exe"
SCRIPT = str(PROJECT_ROOT / "train_histogene_uni_tokens_augmix.py")

BASE_ARGS = [
    "--cross_patient", "--use_augmented_tokens",
    "--lr", "3e-5", "--dropout", "0.5", "--n_encoder_layers", "2",
    "--mixup_alpha", "0.2", "--tv_k", "6",
    "--num_epochs", "150", "--early_stop_patience", "20",
    "--batch_size", "64",
]

FOLDS = {
    1: "JFX+LMZ → HYZ",
    2: "HYZ+LMZ → JFX",
    3: "HYZ+JFX → LMZ",
}


def find_best_from_sweep():
    """从 P0 sweep 结果中找到最佳 (weight, mode) 组合"""
    summary_files = sorted(RESULT_DIR.glob("sweep_summary_*.csv"))
    if not summary_files:
        return None, None

    latest = summary_files[-1]
    best_weight, best_mode, best_pcc = None, None, -999

    with open(latest, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split(',')
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 5:
                continue
            weight, mode, _, pcc_str = parts[0], parts[1], parts[2], parts[3]
            try:
                pcc = float(pcc_str)
                if pcc > best_pcc:
                    best_pcc = pcc
                    best_weight = weight
                    best_mode = mode
            except ValueError:
                continue

    return (best_weight, best_mode, best_pcc) if best_weight else (None, None, None)


def run_fold(fold, tv_weight, tv_mode, timestamp):
    """运行单个 Fold 的训练"""
    run_name = f"TV_3Fold_Fold{fold}_{tv_mode}_w{tv_weight}"
    log_file = CV_RESULT_DIR / f"{run_name}_{timestamp}.log"

    print(f"\n{'─' * 50}")
    print(f"Fold {fold} ({FOLDS[fold]}) 开始 @ {datetime.now().strftime('%H:%M:%S')}")
    print(f"  tv_weight={tv_weight}, tv_mode={tv_mode}")

    cmd = [PYTHON, SCRIPT] + BASE_ARGS + [
        "--fold", str(fold),
        "--tv_weight", str(tv_weight),
        "--tv_mode", tv_mode,
        "--dataset_name", f"TV_3Fold_Fold{fold}",
    ]

    env = {"PYTHONIOENCODING": "utf-8", **__import__("os").environ}

    best_pcc = "ERROR"
    best_loss = "N/A"

    try:
        with open(log_file, "w", encoding="utf-8") as f:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                cwd=str(PROJECT_ROOT),
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                f.write(line)
                f.flush()
                m = re.search(r"\[DONE\]\s*训练结束[。.]\s*最佳 val_loss=([\d.]+),\s*best_pcc=([\d.]+)", line)
                if m:
                    best_loss = m.group(1)
                    best_pcc = m.group(2)
                m2 = re.search(r"方案B-AugMix Best Val PCC:\s*([\d.]+)", line)
                if m2:
                    best_pcc = m2.group(1)

            proc.wait()
            exit_code = proc.returncode

    except Exception as e:
        print(f"  [ERROR] {e}")
        exit_code = -1
        best_pcc = "EXCEPTION"

    print(f"  完成 @ {datetime.now().strftime('%H:%M:%S')} | exit={exit_code} | best_pcc={best_pcc}")
    return {"fold": fold, "pcc": best_pcc, "loss": best_loss, "exit": exit_code}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tv_weight", type=float, default=None, help="TV loss 权重")
    ap.add_argument("--tv_mode", type=str, default=None, choices=["l1", "l2", "laplacian"])
    ap.add_argument("--folds", type=str, default="2,3", help="要运行的 fold，逗号分隔")
    args = ap.parse_args()

    # 确定配置
    tv_weight = args.tv_weight
    tv_mode = args.tv_mode

    if tv_weight is None or tv_mode is None:
        print("[INFO] 未指定 TV 配置，从 P0 sweep 结果自动选择最佳配置...")
        best_weight, best_mode, best_pcc = find_best_from_sweep()
        if best_weight is None:
            print("[ERROR] 未找到 P0 sweep 结果，请先运行 run_tv_sweep.py 或手动指定 --tv_weight/--tv_mode")
            sys.exit(1)
        tv_weight = float(best_weight)
        tv_mode = best_mode
        print(f"[INFO] 自动选择: tv_weight={tv_weight}, tv_mode={tv_mode} (PCC={best_pcc})")
    else:
        print(f"[INFO] 手动指定: tv_weight={tv_weight}, tv_mode={tv_mode}")

    folds_to_run = [int(f.strip()) for f in args.folds.split(",")]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print(f"TV Loss 三折交叉验证")
    print(f"配置: tv_weight={tv_weight}, tv_mode={tv_mode}")
    print(f"Fold 列表: {folds_to_run}")
    print(f"开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = {}
    for fold in folds_to_run:
        results[fold] = run_fold(fold, tv_weight, tv_mode, timestamp)

    # 汇总
    print(f"\n{'=' * 70}")
    print(f"三折交叉验证完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n{'Fold':<6} {'Description':<22} {'Best PCC':<12}")
    print("-" * 45)
    for fold in folds_to_run:
        r = results[fold]
        print(f"{fold:<6} {FOLDS[fold]:<22} {r['pcc']:<12}")

    # 保存 JSON
    summary_path = CV_RESULT_DIR / f"3fold_cv_summary_{timestamp}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "config": {"tv_weight": tv_weight, "tv_mode": tv_mode},
            "results": {str(k): v for k, v in results.items()},
        }, f, indent=2, ensure_ascii=False)
    print(f"\n摘要保存至: {summary_path}")


if __name__ == "__main__":
    main()
