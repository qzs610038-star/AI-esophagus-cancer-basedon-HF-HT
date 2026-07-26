# R4：关键合同与门禁分级

> parent review: `project_state/plans/workflow_governance_v3_refactor_review_20260726.md`
> module id: `R4`
> module revision: `001`
> lifecycle: `pending_review`
> dependencies: `R0`; R3 enforcement waits for this module
> implementation: `not-started`
> boundary: 哈希匹配只证明身份，不自动证明科学有效或接纳结果。

## 1. 目的

只对关键代码闭包、真实输入、训练/选择协议和结论承载结果执行严格指纹；
对可再生文档、日志和无关历史资产降级为 regenerate/diff、inventory 或
WARN，加快调试但不降低实验可复现性。

## 2. Critical Contract Schema

至少包含：

- `experiment_id`
- `protocol_revision`
- `source_commit`
- `critical_code`
- `resolved_hyperparameters`
- `loss_contract`
- `regularization_contract`
- `feature_flags`
- `data_manifest_id`
- `data_manifest_sha256`
- `input_artifacts`
- `selection_policy`
- `external_eval_policy`
- `canonical_sha256`

关键输入还包括实际 checkpoint/base model/calibrator、split、train-only
z-score 和 resolved critical argv。

## 3. Schema 版本拆分

不得继续用一个全局 `SCHEMA_VERSION` 同时控制 state、job、result、repair：

```text
STATE_SCHEMA_VERSION
JOB_SCHEMA_VERSION
RESULT_SCHEMA_VERSION
ASSET_SCHEMA_VERSION
WORKSPACE_SCHEMA_VERSION
```

先拆常量且保持旧行为，再实现 v1/v2 双读和新写 v2。新 v2 source commit
必须为完整 40 位；历史短 commit 只作为 v1 fixture。

## 4. 门禁等级

| 等级 | 对象 | 默认动作 |
|---|---|---|
| E0 | 关键代码、loss、超参、正则、feature、数据/manifest/split/z-score、模型/校准器、选择策略 | 严格 SHA/canonical digest，变化即重新批准 |
| E1 | 路径、解释器、编码、CLI 兼容、日志、回传工程适配 | allowlist + diff + targeted test + adaptation record |
| E2 | Dashboard、用户进度、说明文档、学习卡、可重建图表 | regenerate + diff；不阻断运行，事实冲突时阻断发布 |
| E3 | 历史日志、未引用旧资产、缓存 inventory | WARN/inventory；不参与当前 run |

实际承担选择证据的文件无论原类型如何，都升级为 critical。

## 5. 步骤

1. 定义 canonical JSON 规则。
2. 从最终 resolved 配置提取关键合同，不能只读用户输入参数。
3. active manifest 首次激活全验；run 时只验本 job 引用子集。
4. tracked code 由完整 commit 证明 provenance；非 Git checkpoint/calibrator
   单独哈希。
5. 修改 source hash 键时同步更新 result import 的 staged Registry 和两类
   生成视图事务。
6. 保留 full verification 只读命令供独立审计。

## 6. 风险

- canonicalization 不稳定。
- Python 默认值、隐式 loss 或 launcher alias 未进入合同。
- 引用子集遗漏 group/split/z-score/raw input。
- selection proof 被误分 supporting。
- 过度降级文档门禁掩盖用户视图事实冲突。

## 7. 检验

```powershell
python -m pytest tests/test_critical_contract.py -q
python -m pytest tests/test_mpp_repair_assets.py -q
python -m pytest tests/test_result_import.py -q
```

必须覆盖：

- key 顺序、空白、注释不改变 digest。
- 数值、布尔和有序列表按定义改变 digest。
- 默认参数解析后进入合同。
- job 只验证所引用 MPP group。
- 修改引用输入任一关键 SHA 必须 FAIL。
- 修改未引用 historical 资产不阻断当前 run。
- critical result artifact 被篡改时 import FAIL。
- 用户进度摘要与 Registry 冲突时发布 FAIL，而非训练门禁 FAIL。

## 8. 退出条件

- digest 跨进程和平台稳定。
- 合同覆盖已批准的全部关键选择。
- E0/E1/E2/E3 分类有测试和人工审查清单。
- R3 批准验证真正引用同一 digest。

## 9. 回退

- 保留 full verification 命令。
- feature flag 切回 v1 全量 pre-run verification。
- 不删除已经生成的合同；标记其 schema/version 和未采用状态。

## 10. 待审查

- [ ] 有序/无序列表的 canonicalization 白名单。
- [ ] critical code closure 的提取策略。
- [ ] 大 checkpoint 哈希缓存的失效条件。

## 11. 补充记录

- `2026-07-26 / DIR-20260726-004`：增加用户视图事实冲突与训练门禁的分离。
