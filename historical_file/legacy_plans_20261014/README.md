# 古早方案历史夹(2026-10-14)

**这里的内容仅供追溯,不构成当前规则。** 当前有效规则以仓库根目录 `AGENTS.md`、`project_state/current_state.json` 指向的方案和 `configs/server_paths.yaml` 为准。

## 内容与来源

- 2026-10-14 按《项目瘦身与古早方案降权归档方案》分两批迁移,共 66 份 Markdown(第一批 45 份,解耦批 21 份)。
- 内容是 5–8 月的旧部署方案、旧治理回执(G10–G16)、Gitee 同步与 SSH/远程隧道方案、旧计划笔记,以及 7 月治理 v3 主方案与 17 份分章方案、Ridge 校准旧方案。
- 解耦批同时把 `current_state.json`、`workflow_catalog.json`、`document_registry.json` 里的路径同步到了新位置;`workflow_catalog.json` 里的哈希值未重算。
- 文件**只移动,未改动正文**。子目录沿用原相对路径,例如原 `project_state/governance/X.md` 现在位于 `historical_file/legacy_plans_20261014/project_state/governance/X.md`。

## 查询与还原

- 旧路径到新路径的对照见 [MOVED_MAP.csv](MOVED_MAP.csv)(含移动原因)。
- `project_state/directives.jsonl` 等追溯记录里仍写旧路径,按 `MOVED_MAP.csv` 查新位置。
- 需要还原时,把 `new_path` 的文件移回 `old_path` 即可;`project_state/document_registry.json` 里对应条目的 `path` 同步改回。

## 没有移动的内容(仍在原位)

服务器路径索引、模型部署文档(EGN、HisToGene、OmiCLIP、Loki、Virchow2、UNI2-h)、排坑与排障汇总、`MPP2` 划分与 z-score 协议、评价指标文档、模型获取与预测指南,以及被代码、测试或实验 Registry 引用的文件,均保留在原位。
