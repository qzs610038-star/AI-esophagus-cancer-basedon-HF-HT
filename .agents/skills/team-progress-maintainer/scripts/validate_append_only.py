from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path


TEXT_SUFFIXES = {".md", ".txt", ".csv", ".json", ".yaml", ".yml"}
CORRECTION_MARKER = re.compile(r"补充说明（\d{4}-\d{2}-\d{2}，已获用户审核）")


def collect(path: Path) -> dict[Path, Path]:
    if path.is_file():
        return {Path(path.name): path}
    return {item.relative_to(path): item for item in path.rglob("*") if item.is_file()}


def validate_text(before: Path, after: Path, correction_mode: bool) -> list[str]:
    old = before.read_text(encoding="utf-8").splitlines(keepends=True)
    new = after.read_text(encoding="utf-8").splitlines(keepends=True)
    errors: list[str] = []
    matcher = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in {"delete", "replace"}:
            errors.append(f"检测到旧内容被{('删除' if tag == 'delete' else '改写')}：旧行 {i1 + 1}-{i2}")
        elif tag == "insert" and correction_mode:
            inserted = "".join(new[j1:j2])
            if not CORRECTION_MARKER.search(inserted):
                errors.append(f"修正新增块缺少规定标记：新行 {j1 + 1}-{j2}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="验证维护文件只增加内容，没有改写、删除或移动旧内容。")
    parser.add_argument("before", type=Path, help="修改前快照文件或目录")
    parser.add_argument("after", type=Path, help="修改后文件或目录")
    parser.add_argument("--correction-mode", action="store_true", help="要求每个新增块带审核后的补充说明标记")
    args = parser.parse_args()
    before_root = args.before.resolve()
    after_root = args.after.resolve()
    if not before_root.exists() or not after_root.exists():
        print("FAIL: 修改前快照或修改后路径不存在。")
        return 1

    old_files = collect(before_root)
    new_files = collect(after_root)
    errors: list[str] = []
    checked = 0
    for relative, old_path in old_files.items():
        new_path = new_files.get(relative)
        if new_path is None:
            errors.append(f"旧文件被删除或移动：{relative}")
            continue
        checked += 1
        if old_path.suffix.lower() in TEXT_SUFFIXES:
            for error in validate_text(old_path, new_path, args.correction_mode):
                errors.append(f"{relative}: {error}")
        elif old_path.read_bytes() != new_path.read_bytes():
            errors.append(f"既有二进制文件被改写：{relative}")

    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"PASS: existing_files={checked}, old_content_preserved=yes, correction_mode={args.correction_mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

