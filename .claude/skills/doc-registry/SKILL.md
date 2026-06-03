---
name: doc-registry
description: Document registry for the PFMval project. Use when creating or searching for project documentation. Knows storage paths and enforces naming convention.
argument-hint: [--find <keyword>] [--add <path>] [--list <category>]
allowed-tools: Bash, Read, Write, Glob, Grep
---

# 文档注册表 — 存储规范 + 快速检索

维护 `.claude/doc-registry.json`，统一管理项目文档的存储路径、命名规则和检索。

## 触发规则

| 场景 | 是否触发 | 说明 |
|------|:---:|------|
| 用户要求"创建文档"/"写一份指南"/"记录方案"/"保存分析" | ✅ | 确定路径 + 命名 + 注册 |
| 用户要求"找文档"/"有没有关于X的文档"/"查一下Y方案" | ✅ | 检索注册表，直接定位 |
| Agent/子代理需要定位或查找项目文档 | ✅ | 先查注册表，避免全目录扫描 |
| 创建 memory / 保存经验 / 写入经验到 `.qoder/experience.md` | ✅ | 确定经验记录的正确路径和格式 |
| 日常编码、训练、调试、读代码 | ❌ | 不触发，不读取注册表 |
| 组会汇报、周报 | ✅ | 确定存储到 `02_组会汇报/` |

**关键原则**：仅在文档创建/搜索/定位场景下触发，其他场景不读取 `doc-registry.json` 以节省 token。

---

## 存储路径规则

| 文档内容 | 目标路径 | 命名格式 |
|---------|---------|---------|
| 代码解读、初学者指南、概念说明、迁移教程 | `01_指南与解读/学习指南/` | `{主题}_{YYYYMMDD_HH}.md` |
| 模型部署、集成方案、安装指南、RL 实施方案、执行计划 | `01_指南与解读/部署方案/` | `{模型/方案名}_部署方案_{YYYYMMDD_HH}.md` 或 `{方案名}_{YYYYMMDD_HH}.md` |
| 实验分析、改进建议、文献整理、修复记录、评估报告 | `01_指南与解读/分析报告/` | `{主题}_报告_{YYYYMMDD_HH}.md` 或 `{主题}_{YYYYMMDD_HH}.md` |
| 周报、组会汇报 | `02_组会汇报/` | `组会汇报_{YYYYMMDD}_W{周}周报.md` |
| 项目配置、维护方案、Git 参考 | `.claude/` | 保持现有命名 |
| 文献 PDF、参考文献 | `Ai病理项目文献汇总/` 或 `docs/` | 保持文献原有命名 |
| 一次性分析脚本（Python） | `experiments/analysis/` | `analyze_{主题}_{YYYYMMDD_HH}.py` |
| 扫参/CV 脚本（Python） | `experiments/sweeps/` | `sweep_{主题}_{YYYYMMDD_HH}.py` 或 `run_{主题}_{YYYYMMDD_HH}.py` |
| 实验结果输出（目录） | `experiments/results/{实验名}/` | 自动生成时间戳子目录 |
| 报表图片/图表 | `reports/figures/` | `{报表名}_{日期}.{png,svg}` |
| 旧数据、打包分发、归档 | `archive/` | `{项目名}_打包_{YYYYMMDD_HH}/` 或保持原名 |
| 调试/诊断脚本 | `tools/` | `{debug,diagnose}_{问题}_{YYYYMMDD_HH}.py` |

### 命名细则

- **新建文档**严格使用 `{文档作用}_{YYYYMMDD_HH}.md` 格式，精确到小时
- **已有文档**保留原名，注册表中记录实际文件名和路径
- 示例：`RL训练调度_DQN方案_20260524_22.md`、`AugMix消融实验报告_20260515_14.md`

---

## 检索方式

注册表 `.claude/doc-registry.json` 维护所有文档的结构化索引：

```json
{
  "documents": [
    {
      "name": "RL方法选择_初学者指南.md",
      "path": "01_指南与解读/学习指南/RL方法选择_初学者指南.md",
      "category": "学习指南",
      "purpose": "RL四种方案的可行性分析与选择指南",
      "created": "2026-05-24T22:42",
      "tags": ["RL", "强化学习", "初学者", "方案选择"]
    }
  ]
}
```

### 检索命令

```
/doc-registry --find RL          → 返回所有 tags/purpose/name 含 "RL" 的文档路径
/doc-registry --find 部署        → 返回所有 tags/purpose/name 含 "部署" 的文档路径
/doc-registry --list 学习指南     → 列出学习指南分类下所有文档
/doc-registry --list 部署方案     → 列出部署方案分类下所有文档
/doc-registry --list 分析报告     → 列出分析报告分类下所有文档
/doc-registry --recent           → 最近添加的 10 个文档
/doc-registry --add <path>       → 手动注册一个文档到注册表
```

检索时**只读注册表 JSON**，不扫描目录树，避免 token 浪费。

---

## 创建文档流程

1. **判断分类**：根据文档内容确定存储子目录（学习指南/部署方案/分析报告）
2. **生成文件名**：`{文档作用描述}_{YYYYMMDD_HH}.md`，其中日期时间取当前时刻
3. **写入文件**：将文档内容写入目标路径
4. **注册到注册表**：在 `doc-registry.json` 中追加条目，包含 name/path/category/purpose/created/tags
5. **更新索引**：如果新增了文件类型，同步更新 `01_指南与解读/01_README.md`

### 文档作用（purpose）约定

purpose 字段用一句话描述文档解决什么问题，便于检索时匹配。示例：
- ✅ `"RL 方案可行性分析与选择指南"`
- ✅ `"服务器环境部署完整步骤"`
- ❌ `"一个文档"` — 太模糊，无法检索

---

## 注册表维护

- **存储位置**：`.claude/doc-registry.json`（一个 JSON 文件，方便 grep/jq 处理）
- **更新时机**：每次创建新文档后立即追加条目
- **同步检查**：每月审计时对比注册表与实际文件，清理死链
- **不自动扫描**：注册表通过 `/doc-registry --add` 手动维护，或创建文档时自动追加，避免全目录扫描

---

## 当前项目文档分类速查

| 类别 | 路径 | 典型内容 |
|------|------|---------|
| 学习指南 | `01_指南与解读/学习指南/` | 解读指南、初学者版、迁移教程 |
| 部署方案 | `01_指南与解读/部署方案/` | 部署方案、集成方案、RL 执行策略 |
| 分析报告 | `01_指南与解读/分析报告/` | 实验分析、改进建议、评估报告 |
| 组会汇报 | `02_组会汇报/` | 周报、阶段性汇报 |
| 文献资料 | `Ai病理项目文献汇总/` | 文献 PDF 按主题分文件夹 |
| 实验脚本 | `experiments/` | 分析脚本、扫参脚本、实验结果 |
| 报表 | `reports/` | 图表、图片 |
| 归档 | `archive/` | 旧数据、打包、zip |
| 调试工具 | `tools/` | 诊断脚本 |
| 配置参考 | `.claude/` | 维护方案、Git 参考、agent/skill 定义 |
