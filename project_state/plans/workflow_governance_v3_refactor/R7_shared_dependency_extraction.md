# R7：共享依赖抽取

> parent review: `project_state/plans/workflow_governance_v3_refactor_review_20260726.md`
> module id: `R7`
> module revision: `001`
> lifecycle: `pending_review`
> dependencies: `R6`
> implementation: `not-started`
> boundary: 在血缘与数值兼容测试完成前，不修改受保护目录策略。

## 1. 目的

将仍被 MPP 使用的早期公共能力抽到中性核心模块，解除“旧目录既是历史资产
又承载现役依赖”的混合状态，为后续 lifecycle 迁移和目录解禁创造前提。

首个已知候选：`histogene.utils.compute_metrics`。

## 2. 影响面

- 现役 MPP trainer 的 import；
- 共享 metrics/utility 模块；
- 旧 import compatibility shim；
- pickle/module path 兼容；
- Asset Registry 引用血缘；
- `histogene/` 固定保护边界。

## 3. 步骤

1. 从 R6 血缘中列出全部现役对受保护目录的依赖。
2. 对每个符号建立固定输入/输出兼容 fixture。
3. 在中性核心新增等价实现，先保持行为逐位或数值容差一致。
4. 保留旧 import shim，不立即删除原实现。
5. 逐个迁移 MPP 调用方。
6. 用代码图/import 扫描复核现役调用。
7. Asset Registry 将已迁移符号改为 neutral shared dependency；旧路径仍
   保留 compatibility 生命周期。

## 4. 风险

- metrics 数值漂移。
- NaN、常量、单样本边界行为改变。
- pickle 或动态 import 路径失效。
- 漏掉其他共享符号。
- 过早删除旧入口。

## 5. 检验

```powershell
python -m pytest tests/test_metrics_compatibility.py -q
python -m pytest tests/test_train_mpp_uni2h_mlp.py -q
python -m pytest tests/test_train_mpp_uni2h_lora.py -q
```

必须覆盖：

- 固定输入下新旧 metrics 全字段一致。
- NaN、常量数组、单样本和正常数组边界一致。
- MPP trainer 目标测试通过。
- import/代码图确认现役 MPP 不再直接依赖受保护目录。
- compatibility shim 有弃用提示但仍可用。

## 6. 退出条件

- R6 识别的现役共享依赖全部有处理结论。
- 数值兼容测试通过。
- 新旧 import 迁移路径可回退。
- 不以“代码可导入”代替运行测试。

## 7. 回退

MPP 调用方切回旧 import；保留新模块但不启用。旧实现和 compatibility
shim 不删除。

## 8. 待审查

- [ ] 中性核心模块最终命名与包路径。
- [ ] 数值等价采用逐位还是明确容差。
- [ ] compatibility shim 保留周期。

## 9. 补充记录

- `2026-07-26 / DIR-20260726-004`：首次拆分为独立阶段文件。
