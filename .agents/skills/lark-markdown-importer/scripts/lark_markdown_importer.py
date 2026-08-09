#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lark_markdown_importer.py
通用飞书 Markdown 文档精准导入与图片按原章节定位对齐工具。
支持自动处理图片、处理 HTML details 标签、建立结构锚点与 Block ID 联动定位。
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

def import_markdown_to_lark(doc_path, title=None):
    doc_path = os.path.abspath(doc_path)
    if not os.path.exists(doc_path):
        print(json.dumps({"ok": False, "error": f"File not found: {doc_path}"}, ensure_ascii=False))
        return

    doc_dir = os.path.dirname(doc_path)
    filename = os.path.basename(doc_path)
    doc_title = title or os.path.splitext(filename)[0]

    with open(doc_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. 查找所有图片匹配 ![alt](rel_path)
    img_matches = list(re.finditer(r'!\[(.*?)\]\((.*?)\)', content))
    
    # 2. 预处理 Markdown：剥离 <details> 等易混淆标签，替换图片为唯一位置锚点
    processed_content = content.replace("<details>", "").replace("</details>", "")
    # 替换 <summary> 标签为 Markdown H3
    processed_content = re.sub(r'<summary><strong>(.*?)</strong></summary>', r'### \1', processed_content)
    processed_content = re.sub(r'<summary>(.*?)</summary>', r'### \1', processed_content)

    img_anchors = []
    for idx, match in enumerate(img_matches):
        alt = match.group(1).strip() or f"图片_{idx+1}"
        img_rel_path = match.group(2).strip()
        anchor_text = f"【位置锚点-{idx+1:02d}: {alt}】"
        img_anchors.append((img_rel_path, anchor_text, alt))

        # 将 Markdown 中的原图片标签替换为锚点段落
        pattern = re.escape(match.group(0))
        processed_content = re.sub(pattern, f"\n\n**{anchor_text}**\n\n", processed_content, count=1)

    # 3. 写入临时 markdown 文件（当前目录相对路径）
    temp_md_path = "./temp_lark_import_doc.md"
    with open(temp_md_path, "w", encoding="utf-8") as tf:
        tf.write(processed_content)

    # 4. 创建初始文档
    create_res = run_lark_cli(["docs", "+create", "--title", doc_title, "--doc-format", "markdown", "--content", "@./temp_lark_import_doc.md"])
    
    if os.path.exists(temp_md_path):
        os.remove(temp_md_path)

    if not create_res.get("ok"):
        print(json.dumps({"ok": False, "error": "Failed to create document", "detail": create_res}, ensure_ascii=False))
        return

    doc_info = create_res["data"]["document"]
    doc_id = doc_info["document_id"]
    doc_url = doc_info["url"]

    # 5. 拉取 XML 结构以搜索块 ID
    fetch_res = run_lark_cli(["docs", "+fetch", "--doc", doc_id, "--detail", "with-ids"])
    if not fetch_res.get("ok"):
        print(json.dumps({"ok": False, "error": "Failed to fetch document block IDs", "detail": fetch_res}, ensure_ascii=False))
        return

    xml_content = fetch_res["data"]["document"]["content"]

    # 6. 逐个定位锚点并上移图片
    results = []
    for idx, (img_rel_path, anchor_text, alt) in enumerate(img_anchors):
        # 计算图片绝对与相对路径
        full_img_path = os.path.normpath(os.path.join(doc_dir, img_rel_path))
        if not os.path.exists(full_img_path):
            results.append({"image": img_rel_path, "status": "skipped", "reason": "Image file not found on disk"})
            continue

        rel_img_path = "./" + os.path.relpath(full_img_path, start=os.getcwd()).replace("\\", "/")

        # 查找锚点文本所在的 Block ID
        pos = xml_content.find(anchor_text)
        if pos == -1:
            results.append({"image": img_rel_path, "status": "failed", "reason": f"Anchor text '{anchor_text}' not found in XML"})
            continue

        sub_xml = xml_content[:pos]
        id_matches = list(re.finditer(r'id=["\']([^"\']+)["\']', sub_xml))
        if not id_matches:
            results.append({"image": img_rel_path, "status": "failed", "reason": "Could not extract block ID for anchor"})
            continue

        anchor_block_id = id_matches[-1].group(1)

        # 插入图片
        insert_res = run_lark_cli(["docs", "+media-insert", "--doc", doc_id, "--file", rel_img_path, "--align", "center", "--caption", alt])
        if not insert_res.get("ok"):
            results.append({"image": img_rel_path, "status": "failed", "reason": "media-insert failed", "detail": insert_res})
            continue

        img_block_id = insert_res["data"]["block_id"]

        # 移动图片至对应章节锚点 Block 之后
        move_res = run_lark_cli(["docs", "+update", "--doc", doc_id, "--command", "block_move_after", "--block-id", anchor_block_id, "--src-block-ids", img_block_id])
        if move_res.get("ok"):
            results.append({"image": img_rel_path, "status": "success", "block_id": img_block_id, "anchor_block_id": anchor_block_id})
        else:
            results.append({"image": img_rel_path, "status": "partial_move_failed", "detail": move_res})

    output = {
        "ok": True,
        "document_id": doc_id,
        "url": doc_url,
        "total_images": len(img_anchors),
        "image_results": results
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))

def main():
    parser = argparse.ArgumentParser(description="Precision Markdown to Feishu Doc Importer with Inline Image Alignment")
    parser.add_argument("--file", "-f", required=True, help="Path to local markdown file")
    parser.add_argument("--title", "-t", help="Target Feishu document title (default: filename)")
    args = parser.parse_args()

    import_markdown_to_lark(args.file, args.title)

if __name__ == "__main__":
    main()
