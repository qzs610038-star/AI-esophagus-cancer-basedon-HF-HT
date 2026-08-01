from __future__ import annotations

import argparse
import re
import sys
import urllib.parse
from pathlib import Path


def without_fenced_code(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def main() -> int:
    parser = argparse.ArgumentParser(description="校验 qzs 稳定结论文档结构、表述和本地证据链接。")
    parser.add_argument("document", type=Path)
    args = parser.parse_args()
    document = args.document.resolve()
    if not document.is_file():
        print(f"FAIL: 文档不存在：{document}")
        return 1

    text = document.read_text(encoding="utf-8")
    errors: list[str] = []
    if text.count("<details>") != text.count("</details>"):
        errors.append("<details> 标签未完整闭合")

    chunks = [match.group(0) for match in re.finditer(r"(?ms)^## (?!一、).*?(?=^## |\Z)", text)]
    experiment_chunks = [chunk for chunk in chunks if "<details>" in chunk]
    for index, chunk in enumerate(experiment_chunks, start=1):
        heading = chunk.splitlines()[0]
        if chunk.count("<details>") != 4 or chunk.count("</details>") != 4:
            errors.append(f"{heading}: 必须有且仅有四个折叠部分")
        for number in range(1, 5):
            if len(re.findall(fr"<summary>.*?{number}\.", chunk)) != 1:
                errors.append(f"{heading}: 缺少或重复第 {number} 部分")
        if not re.search(r"\*\*实验完成日期：.+?\*\*", chunk):
            errors.append(f"{heading}: 缺少实验完成日期")
        if "中文逻辑" not in chunk or "English pseudocode" not in chunk:
            errors.append(f"{heading}: 缺少中英文对照伪代码")
        if not re.search(r"\([^)]*\.py#L\d+\)", chunk):
            errors.append(f"{heading}: 缺少带行号的真实 Python 实现链接")

    prose = without_fenced_code(text)
    for banned in ("门控", "合同", "NO-GO"):
        if banned in prose:
            errors.append(f"正文含未展开内部术语：{banned}")

    link_pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    local_links = 0
    image_links = 0
    for raw_target in link_pattern.findall(text):
        target = raw_target.strip().strip("<>")
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        clean = urllib.parse.unquote(target.split("#", 1)[0])
        if not clean:
            continue
        local_links += 1
        resolved = (document.parent / clean).resolve()
        if not resolved.exists():
            errors.append(f"失效本地链接：{target}")
        if raw_target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
            image_links += 1
            if "可视化图表" not in clean.replace("\\", "/"):
                errors.append(f"图表未归档到 qzs 可视化图表目录：{target}")

    if not experiment_chunks:
        errors.append("没有识别到实验章节")
    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print(
        f"PASS: experiments={len(experiment_chunks)}, details={text.count('<details>')}, "
        f"local_links={local_links}, images={image_links}, banned_terms=0"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

