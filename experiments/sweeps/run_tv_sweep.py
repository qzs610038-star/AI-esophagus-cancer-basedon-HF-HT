"""
TV Loss 超参数扫描执行器
9 组合: w ∈ {0.01, 0.05, 0.1} × mode ∈ {l1, l2, laplacian}
使用 cross-patient Fold1 (JFX+LMZ → HYZ) + AugMix 基线配置
"""
import subprocess
import sys
import re
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RESULT_DIR = PROJECT_ROOT / "tv_sweep_results"
RESULT_DIR.mkdir(exist_ok=True)

PYTHON = r"C:\Program Files\Python313\python.exe"
SCRIPT = str(PROJECT_ROOT / "train_histogene_uni_tokens_augmix.py")

# 固定参数（匹配最佳基线: PCC=0.4212 @ tv_weight=0.05, mode=l1）
BASE_ARGS = [
    "--cross_patient", "--fold", "1", "--use_augmented_tokens",
    "--lr", "3e-5", "--dropout", "0.5", "--n_encoder_layers", "2",
    "--mixup_alpha", "0.2", "--tv_k", "6",
    "--num_epochs", "150", "--early_stop_patience", "20",
    "--batch_size", "64",
]

WEIGHTS = ["0.01", "0.05", "0.1"]
MODES = ["l1", "l2", "laplacian"]

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SUMMARY_FILE = RESULT_DIR / f"sweep_summary_{TIMESTAMP}.csv"

def main():
    results = []
    total = len(WEIGHTS) * len(MODES)
    counter = 0

    print("=" * 70)
    print(f"TV Loss 超参数扫描 — {total} 组合")
    print(f"开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"摘要文件: {SUMMARY_FILE}")
    print("=" * 70)

    for weight in WEIGHTS:
        for mode in MODES:
            counter += 1
            run_name = f"tv_{mode}_w{weight}"
            log_file = RESULT_DIR / f"{run_name}_{TIMESTAMP}.log"

            print(f"\n{'─' * 50}")
            print(f"[{counter}/{total}] {run_name} 开始 @ {datetime.now().strftime('%H:%M:%S')}")
            print(f"  日志: {log_file.name}")

            cmd = [PYTHON, SCRIPT] + BASE_ARGS + [
                "--tv_weight", weight,
                "--tv_mode", mode,
                "--dataset_name", f"TV_Sweep_{run_name}",
            ]

            env = {"PYTHONIOENCODING": "utf-8", **__import__("os").environ}

            best_pcc = "ERROR"
            best_epoch = "N/A"
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
                        # 捕获结果行
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

            print(f"  完成 @ {datetime.now().strftime('%H:%M:%S')} | exit={exit_code} | best_pcc={best_pcc} | epoch_best_loss={best_loss}")

            results.append({
                "weight": weight,
                "mode": mode,
                "run_name": run_name,
                "best_pcc": best_pcc,
                "best_loss": best_loss,
                "exit_code": exit_code,
                "log_file": str(log_file),
            })

    # 写 CSV 摘要
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write("weight,mode,run_name,best_pcc,best_loss,exit_code,log_file\n")
        for r in results:
            f.write(f"{r['weight']},{r['mode']},{r['run_name']},{r['best_pcc']},{r['best_loss']},{r['exit_code']},{r['log_file']}\n")

    print(f"\n{'=' * 70}")
    print(f"扫描完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n快速汇总:")
    print(f"{'Weight':<8} {'Mode':<12} {'Best PCC':<12} {'Exit'}")
    print("-" * 45)
    for r in results:
        print(f"{r['weight']:<8} {r['mode']:<12} {r['best_pcc']:<12} {r['exit_code']}")
    print(f"\n完整结果: {SUMMARY_FILE}")

    # 找出最佳
    valid = [(r['weight'], r['mode'], r['best_pcc']) for r in results
             if r['best_pcc'] not in ("ERROR", "EXCEPTION")]
    if valid:
        best = max(valid, key=lambda x: float(x[2]))
        print(f"\n★★★ 最佳配置: tv_weight={best[0]}, tv_mode={best[1]}, PCC={best[2]} ★★★")

if __name__ == "__main__":
    main()
