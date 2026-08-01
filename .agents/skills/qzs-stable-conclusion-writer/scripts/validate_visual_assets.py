from __future__ import annotations

import argparse
import hashlib
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="校验 qzs 图表可读性、PNG/SVG 配对和可选的权威原图哈希。")
    parser.add_argument("visualization_dir", type=Path)
    parser.add_argument("--source-dir", type=Path, help="可选：含权威原图的目录；比较同名 PNG/SVG")
    args = parser.parse_args()
    root = args.visualization_dir.resolve()
    if not root.is_dir():
        print(f"FAIL: 图表目录不存在：{root}")
        return 1

    try:
        from PIL import Image
    except ImportError:
        print("FAIL: 缺少 Pillow，无法验证 PNG 可读性。")
        return 1

    pngs = sorted(root.rglob("*.png"))
    svgs = sorted(root.rglob("*.svg"))
    errors: list[str] = []
    for path in pngs:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"PNG 无法读取：{path.relative_to(root)} ({exc})")
        if not path.with_suffix(".svg").exists():
            errors.append(f"PNG 缺少同名 SVG：{path.relative_to(root)}")
    for path in svgs:
        try:
            ET.parse(path)
        except ET.ParseError as exc:
            errors.append(f"SVG 无法解析：{path.relative_to(root)} ({exc})")
        if not path.with_suffix(".png").exists():
            errors.append(f"SVG 缺少同名 PNG：{path.relative_to(root)}")

    compared = 0
    if args.source_dir:
        source = args.source_dir.resolve()
        if not source.is_dir():
            errors.append(f"权威原图目录不存在：{source}")
        else:
            source_by_name = {p.name: p for p in source.rglob("*") if p.suffix.lower() in {".png", ".svg"}}
            for destination in pngs + svgs:
                original = source_by_name.get(destination.name)
                if original:
                    compared += 1
                    if digest(original) != digest(destination):
                        errors.append(f"复制图与权威原图哈希不一致：{destination.name}")

    if not pngs or not svgs:
        errors.append("至少需要一组 PNG 和 SVG 图表")
    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"PASS: png={len(pngs)}, svg={len(svgs)}, paired=yes, source_hash_matches={compared}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

