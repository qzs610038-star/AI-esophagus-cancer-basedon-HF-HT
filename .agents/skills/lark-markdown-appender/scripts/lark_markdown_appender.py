#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lark_markdown_appender.py
飞书云文档追加式补充工具：在已有飞书云文档后追加 Markdown 内容，并精准对齐新追加的内嵌图片到对应新章节下方。
"""

import os
import sys
import re
import json
import argparse
import subprocess

def run_lark_cli(cmd_args):
    cmd = ["lark-cli"] + cmd_args
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", shell=True)
    if res.returncode != 0:
        try:
            return json.loads(res.stdout)
        except:
            return {"ok": False, "error": res.stderr or res.stdout}
    try:
        return json.loads(res.stdout)
    except:
        return {"ok": True, "raw": res.stdout}

def extract_doc_id(doc_input):
    if not doc_input:
        return None
    # 支持形如 https://xxx.feishu.cn/docx/doxcnXXXX 或直接 doxcnXXXX
    match = re.search(r'/docx/([a-zA-Z0-9]+)', doc_input)
    if match:
        return match.group(1)
    return doc_input.strip()

def append_markdown_to_lark(doc_input, doc_path, target_block_id="-1"):
    doc_id = extract_doc_id(doc_input)
    if not doc_id:
        print(json.dumps({"ok": False, "error": f"Invalid doc input: {doc_input}"}, ensure_ascii=False))
        return

    doc_path = os.path.abspath(doc_path)
    if not os.path.exists(doc_path):
        print(json.dumps({"ok": False, "error": f"File not found: {doc_path}"}, ensure_ascii=False))
        return

    doc_dir = os.path.dirname(doc_path)

    with open(doc_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. 查找所有追加图片匹配 ![alt](rel_path)
    img_matches = list(re.finditer(r'!\[(.*?)\]\((.*?)\)', content))
    
    # 2. 预处理待追加的 Markdown：剥离易混淆 HTML，标记唯一位置锚点
    processed_content = content.replace("<details>", "").replace("</details>", "")
    processed_content = re.sub(r'<summary><strong>(.*?)</strong></summary>', r'### \1', processed_content)
    processed_content = re.sub(r'<summary>(.*?)</summary>', r'### \1', processed_content)

    img_anchors = []
    timestamp_prefix = f"APP_{os.getpid()}"
    for idx, match in enumerate(img_matches):
        alt = match.group(1).strip() or f"追加图片_{idx+1}"
        img_rel_path = match.group(2).strip()
        anchor_text = f"【追加锚点-{timestamp_prefix}-{idx+1:02d}: {alt}】"
        img_anchors.append((img_rel_path, anchor_text, alt))

        pattern = re.escape(match.group(0))
        processed_content = re.sub(pattern, f"\n\n**{anchor_text}**\n\n", processed_content, count=1)

    # 3. 写入临时 markdown 文件
    temp_md_path = "./temp_lark_append_doc.md"
    with open(temp_md_path, "w", encoding="utf-8") as tf:
        tf.write(processed_content)

    # 4. 执行追加指令 (append 或 block_insert_after)
    if target_block_id == "-1":
        append_res = run_lark_cli(["docs", "+update", "--doc", doc_id, "--command", "append", "--doc-format", "markdown", "--content", "@./temp_lark_append_doc.md"])
    else:
        append_res = run_lark_cli(["docs", "+update", "--doc", doc_id, "--command", "block_insert_after", "--block-id", target_block_id, "--doc-format", "markdown", "--content", "@./temp_lark_append_doc.md"])

    if os.path.exists(temp_md_path):
        os.remove(temp_md_path)

    if not append_res.get("ok"):
        print(json.dumps({"ok": False, "error": "Failed to append content to document", "detail": append_res}, ensure_ascii=False))
        return

    # 5. 拉取全文档 XML 结构以定位新追加块
    fetch_res = run_lark_cli(["docs", "+fetch", "--doc", doc_id, "--detail", "with-ids"])
    if not fetch_res.get("ok"):
        print(json.dumps({"ok": False, "error": "Failed to fetch document block IDs after append", "detail": fetch_res}, ensure_ascii=False))
        return

    xml_content = fetch_res["data"]["document"]["content"]

    # 6. 逐个定位新追加锚点并移动图片
    results = []
    for idx, (img_rel_path, anchor_text, alt) in enumerate(img_anchors):
        full_img_path = os.path.normpath(os.path.join(doc_dir, img_rel_path))
        if not os.path.exists(full_img_path):
            results.append({"image": img_rel_path, "status": "skipped", "reason": "Image file not found on disk"})
            continue

        rel_img_path = "./" + os.path.relpath(full_img_path, start=os.getcwd()).replace("\\", "/")

        pos = xml_content.find(anchor_text)
        if pos == -1:
            results.append({"image": img_rel_path, "status": "failed", "reason": f"Anchor text '{anchor_text}' not found"})
            continue

        sub_xml = xml_content[:pos]
        id_matches = list(re.finditer(r'id=["\']([^"\']+)["\']', sub_xml))
        if not id_matches:
            results.append({"image": img_rel_path, "status": "failed", "reason": "Could not extract block ID for anchor"})
            continue

        anchor_block_id = id_matches[-1].group(1)

        # 插入新图片
        insert_res = run_lark_cli(["docs", "+media-insert", "--doc", doc_id, "--file", rel_img_path, "--align", "center", "--caption", alt])
        if not insert_res.get("ok"):
            results.append({"image": img_rel_path, "status": "failed", "reason": "media-insert failed", "detail": insert_res})
            continue

        img_block_id = insert_res["data"]["block_id"]

        # 移动新图片到追加的对应锚点 Block 之后
        move_res = run_lark_cli(["docs", "+update", "--doc", doc_id, "--command", "block_move_after", "--block-id", anchor_block_id, "--src-block-ids", img_block_id])
        if move_res.get("ok"):
            results.append({"image": img_rel_path, "status": "success", "block_id": img_block_id, "anchor_block_id": anchor_block_id})
        else:
            results.append({"image": img_rel_path, "status": "partial_move_failed", "detail": move_res})

    output = {
        "ok": True,
        "document_id": doc_id,
        "appended_images": len(img_anchors),
        "image_results": results
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))

def main():
    parser = argparse.ArgumentParser(description="Precision Markdown Appender for Feishu Documents")
    parser.add_argument("--doc", "-d", required=True, help="Target Feishu document ID or URL")
    parser.add_argument("--file", "-f", required=True, help="Path to markdown file to append")
    parser.add_argument("--block-id", "-b", default="-1", help="Target block ID to append after (default: -1 for end of document)")
    args = parser.parse_args()

    append_markdown_to_lark(args.doc, args.file, args.block_id)

if __name__ == "__main__":
    main()
