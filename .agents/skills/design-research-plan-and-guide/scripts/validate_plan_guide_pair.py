#!/usr/bin/env python3
"""校验部署方案与学习指南的关键决策编号和深链。"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


KDP_HEADING = re.compile(r"^#{2,4}\s+KDP-([1-4])(?:\s*[:：]|\s+)", re.MULTILINE | re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验部署方案与学习指南是否成对一致")
    parser.add_argument("--plan", required=True, type=Path, help="部署方案 Markdown 路径")
    parser.add_argument("--guide", required=True, type=Path, help="学习指南 Markdown 路径")
    parser.add_argument("--required-term", action="append", default=[], help="要求同时出现的术语；可重复传入")
    return parser.parse_args()


def ordered_kdps(text: str) -> list[str]:
    return [match.group(1) for match in KDP_HEADING.finditer(text)]


def validate(plan: str, guide: str, required_terms: list[str]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    plan_kdps = ordered_kdps(plan)
    guide_kdps = ordered_kdps(guide)

    if not 2 <= len(plan_kdps) <= 4:
        errors.append(f"部署方案必须包含 2–4 个 KDP 标题，当前为 {len(plan_kdps)} 个")
    if len(plan_kdps) != len(set(plan_kdps)):
        errors.append("部署方案存在重复 KDP 编号")
    if plan_kdps != guide_kdps:
        errors.append(f"KDP 编号或顺序不一致：plan={plan_kdps}, guide={guide_kdps}")

    first_detail_heading = re.search(r"^##\s+(?:1[.、\s]|一[、.\s])", plan, re.MULTILINE)
    top_limit = first_detail_heading.start() if first_detail_heading else min(len(plan), 12000)
    for number in plan_kdps:
        if not re.search(rf"\]\([^\n)]*#kdp-{number}\)", plan[:top_limit], re.IGNORECASE):
            errors.append(f"部署方案顶部缺少指向指南 #kdp-{number} 的深链")
        if not re.search(rf"<a\s+id=[\"']kdp-{number}[\"']\s*></a>", guide, re.IGNORECASE):
            errors.append(f"学习指南缺少显式锚点 <a id=\"kdp-{number}\"></a>")

    for term in required_terms:
        missing = [name for name, text in (("部署方案", plan), ("学习指南", guide)) if term not in text]
        if missing:
            errors.append(f"术语 {term!r} 未出现在：{'、'.join(missing)}")

    if "非目标" not in plan:
        warnings.append("部署方案未发现“非目标”章节")
    if "防泄漏" not in plan and "数据泄漏" not in plan:
        warnings.append("部署方案未发现防泄漏边界")
    if "风险" not in guide:
        warnings.append("学习指南未发现风险说明")
    if "用户决策" not in guide and "用户审核" not in guide:
        warnings.append("学习指南未发现用户决策或审核部分")

    return errors, warnings


def main() -> int:
    args = parse_args()
    plan = args.plan.read_text(encoding="utf-8")
    guide = args.guide.read_text(encoding="utf-8")
    errors, warnings = validate(plan, guide, args.required_term)
    for message in warnings:
        print(f"[WARN] {message}")
    for message in errors:
        print(f"[ERROR] {message}")
    if errors:
        print(f"[SUMMARY] ERROR={len(errors)} WARN={len(warnings)}")
        return 1
    print(f"[PASS] 成对文档校验通过；WARN={len(warnings)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
