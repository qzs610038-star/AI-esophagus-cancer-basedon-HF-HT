# Phase 3 CLAM 基线参考实现来源说明

## 管理身份

- 正式管理路径：`reference_implementations/phase3_clam_escc_20260801/`
- 备份来源：`团队项目进度与结论/yzq/0801-CLAM/`
- 纳入日期：2026-08-03
- 生命周期：`reference_candidate`（待本项目核验的参考实现）
- 证据边界：本目录的存在只证明代码、配置和切分已纳入版本化管理，不证明训练已运行、结果有效或方案已获实验执行批准。

## 复制范围

- 来源文件总数：300。
- 正式副本文件数：275（不含本说明文件）。
- 保留：Python 源码、环境文件、数据表、切分文件、配置、上游说明、许可证和文档图片。
- 排除：22 个 Python 字节码缓存；`heatmaps/demo/` 中 2 张示例 SVS 和 1 个示例 checkpoint。
- 排除理由：缓存不是源码；示例切片和 checkpoint 是上游演示二进制，不属于本项目 Phase 3 基线协议，也不应通过项目代码通道传播。

## 上游与许可证

- 基础代码来源：[Mahmood Lab CLAM](https://github.com/mahmoodlab/CLAM)。
- CLAM 代码随包保留 `LICENSE.md`（GPLv3）。
- 当前项目输入写明使用 UNI2-h 预提取特征；UNI2-h 的模型与衍生物受其[官方模型卡](https://huggingface.co/MahmoodLab/UNI2-h)所列非商业研究和不可再分发条款约束。本目录未复制 UNI2-h 权重或特征文件。

## 当前使用边界

1. 不把本目录直接当作 active 实验实现；后续实验仍需遵守 `W###` 工作树、实验登记和用户批准门禁。
2. 不运行训练、不导入结果、不接纳性能结论。
3. 先解决标签病理定义、MPP2 输入究竟为 accepted raw output 还是现有 `z_scores`、主评估层级及 pCR/MPR 共用切分入口等问题。
4. 详细审查见 `01_指南与解读/分析报告/Phase3_CLAM基线方案审查与文献注册_20260803.md`。
