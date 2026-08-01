from __future__ import annotations

import argparse
import re
import sys
import urllib.parse
from pathlib import Path


REQUIRED_DIRS = (
    "yzq",
    "wzk",
    "ljq",
    "qzs",
    "lzd",
    "汇总",
    "汇总/Phase2与Phase3",
    "汇总/论文写作",
    "汇总/训练数据",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="校验团队进度目录结构和 Markdown 本地链接。")
    parser.add_argument("team_root", type=Path)
    args = parser.parse_args()
    root = args.team_root.resolve()
    errors: list[str] = []

    if not root.is_dir():
        print(f"FAIL: 目录不存在：{root}")
        return 1

    for relative in REQUIRED_DIRS:
        if not (root / relative).is_dir():
            errors.append(f"缺少目录：{relative}")
    for relative in ("README.md", "英文专业术语速查表.md"):
        if not (root / relative).is_file():
            errors.append(f"缺少文件：{relative}")

    link_count = 0
    link_pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    for markdown in root.rglob("*.md"):
        text = markdown.read_text(encoding="utf-8")
        for raw_target in link_pattern.findall(text):
            target = raw_target.strip().strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            clean = urllib.parse.unquote(target.split("#", 1)[0])
            if not clean:
                continue
            link_count += 1
            if not (markdown.parent / clean).resolve().exists():
                errors.append(f"失效链接：{markdown.relative_to(root)} -> {target}")

    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"PASS: required_dirs={len(REQUIRED_DIRS)}, local_links={link_count}, missing=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())

