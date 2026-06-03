#!/bin/bash
# TV Loss 超参数扫描脚本
# 9 组合: w ∈ {0.01, 0.05, 0.1} × mode ∈ {l1, l2, laplacian}
# 使用 cross-patient Fold1 (JFX+LMZ → HYZ) + AugMix 基线配置
# 预计总耗时: ~2.5h (每轮 ~17min)

set -e

PYTHON="C:/Program Files/Python313/python.exe"
SCRIPT="train_histogene_uni_tokens_augmix.py"
RESULT_DIR="tv_sweep_results"
SUMMARY_FILE="${RESULT_DIR}/sweep_summary_$(date +%Y%m%d_%H%M%S).txt"

# 固定参数 (匹配最佳基线: PCC=0.4212 @ tv_weight=0.05, mode=l1)
BASE_ARGS="--cross_patient --fold 1 --use_augmented_tokens \
  --lr 3e-5 --dropout 0.5 --n_encoder_layers 2 \
  --mixup_alpha 0.2 --tv_k 6 \
  --num_epochs 150 --early_stop_patience 20 \
  --batch_size 64"

mkdir -p "${RESULT_DIR}"

echo "============================================================" | tee -a "${SUMMARY_FILE}"
echo "TV Loss 超参数扫描" | tee -a "${SUMMARY_FILE}"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${SUMMARY_FILE}"
echo "固定参数: ${BASE_ARGS}" | tee -a "${SUMMARY_FILE}"
echo "============================================================" | tee -a "${SUMMARY_FILE}"
echo "" | tee -a "${SUMMARY_FILE}"

COUNTER=0
TOTAL=9

for WEIGHT in 0.01 0.05 0.1; do
  for MODE in l1 l2 laplacian; do
    COUNTER=$((COUNTER + 1))
    RUN_NAME="tv_${MODE}_w${WEIGHT}"

    echo "" | tee -a "${SUMMARY_FILE}"
    echo "──────── [${COUNTER}/${TOTAL}] ${RUN_NAME} ────────" | tee -a "${SUMMARY_FILE}"
    echo "开始: $(date '+%H:%M:%S')" | tee -a "${SUMMARY_FILE}"

    LOG_FILE="${RESULT_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

    # 实际训练命令
    PYTHONIOENCODING=utf-8 "${PYTHON}" "${SCRIPT}" \
      ${BASE_ARGS} \
      --tv_weight ${WEIGHT} \
      --tv_mode ${MODE} \
      --dataset_name "TV_Sweep_${RUN_NAME}" \
      2>&1 | tee "${LOG_FILE}"

    EXIT_CODE=$?

    # 提取最佳 PCC
    BEST_PCC=$(grep -oP "Best Test PCC:\s*\K[0-9.]+" "${LOG_FILE}" 2>/dev/null || echo "N/A")
    BEST_EPOCH=$(grep -oP "Best Epoch:\s*\K[0-9]+" "${LOG_FILE}" 2>/dev/null || echo "N/A")

    echo "完成: $(date '+%H:%M:%S') | Exit=${EXIT_CODE} | Best PCC=${BEST_PCC} @ Epoch=${BEST_EPOCH}" | tee -a "${SUMMARY_FILE}"

  done
done

echo "" | tee -a "${SUMMARY_FILE}"
echo "============================================================" | tee -a "${SUMMARY_FILE}"
echo "扫描完成: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${SUMMARY_FILE}"
echo "结果摘要: ${SUMMARY_FILE}" | tee -a "${SUMMARY_FILE}"
echo "============================================================" | tee -a "${SUMMARY_FILE}"

# 提取所有结果汇总
echo "" | tee -a "${SUMMARY_FILE}"
echo "──────── 快速汇总 ────────" | tee -a "${SUMMARY_FILE}"
grep -E "^\ ────────|完成:" "${SUMMARY_FILE}" | tee -a "${SUMMARY_FILE}"
