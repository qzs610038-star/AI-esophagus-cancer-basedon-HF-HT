---
name: lark-markdown-appender
version: 1.0.0
description: "飞书云文档追加式补充技能：在已有飞书云文档结尾或指定 Block 下追加本地 Markdown 内容，同时自动提取新增 Markdown 中的图片并精准移入追加章节原位，保留已有正文与样式不受损坏。"
metadata:
  requires:
    bins: ["lark-cli", "python"]
    skills: ["lark-doc", "lark-drive"]
---

# Lark Markdown Appender Skill (追加补充技能)

本技能提供向**已存在的飞书云文档**进行增量追加、补充更新的完整自动化支持。

## 🎯 核心解决痛点

1. **原文档保护与增量追加**：无需全量覆盖或重新创建文档，保留原有文档的所有结构、评论与历史版本，仅将新的 Markdown 补充到文档结尾或指定章节 Block 下。
2. **新增图片按章节精准原位插入**：在追加新 Markdown 内容时，自动识别新增 Markdown 中的本地/相对路径图片，并使用结构化锚点精确将新图片嵌入回新增章节对应的段落下方。

---

## 🛠️ CLI & 脚本使用方式

项目内置可复用 Python 追加工具：
脚本路径：[scripts/lark_markdown_appender.py](scripts/lark_markdown_appender.py)

### 1. 命令行直接调用

```bash
python .agents/skills/lark-markdown-appender/scripts/lark_markdown_appender.py --doc <目标云文档URL或ID> --file <待追加的Markdown文件> [--block-id <指定Block_ID>]
```

### 2. 参数说明

| 参数 | 必填 | 默认值 | 说明 |
| :--- | :---: | :---: | :--- |
| `--doc`, `-d` | 是 | - | 目标飞书云文档的 URL 或 token/ID（如 `https://my.feishu.cn/docx/P4hUd...`） |
| `--file`, `-f` | 是 | - | 包含增量内容的本地 Markdown 文件路径 |
| `--block-id`, `-b` | 否 | `-1` | 目标追加位置 Block ID（默认 `-1` 表示追加到文档最末尾；也可指定特定 Block ID） |

### 3. 输出格式

执行完毕后输出标准 JSON 格式：

```json
{
  "ok": true,
  "document_id": "P4hUdifeKovimHx7K0Dck3cinsl",
  "appended_images": 3,
  "image_results": [
    {
      "image": "images/new_chart.png",
      "status": "success",
      "block_id": "doxcn...",
      "anchor_block_id": "doxcn..."
    }
  ]
}
```

---

## 💡 其他 Agent 使用本 Skill 指引

当需要对现有的项目进展、组会记录或实验总结文档进行**增量更新或追加补发**时，直接调用本 Skill：

```powershell
python .agents/skills/lark-markdown-appender/scripts/lark_markdown_appender.py --doc "https://my.feishu.cn/docx/你的文档ID" --file "补充内容.md"
```
