import re
import os
import subprocess
import json

DOC_PATH = r"D:\AI空间转录病理研究\PFMval_new\团队项目进度与结论\qzs\MPP2基线修复与后续实验进展_20260731.md"
DOC_DIR = os.path.dirname(DOC_PATH)

def run_lark_cli(cmd_args):
    cmd = ["lark-cli"] + cmd_args
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", shell=True)
    if res.returncode != 0:
        print(f"Error running {' '.join(cmd)}: {res.stderr}")
        try:
            return json.loads(res.stdout)
        except:
            return {"ok": False, "error": res.stderr}
    try:
        return json.loads(res.stdout)
    except:
        return {"ok": True, "raw": res.stdout}

def main():
    print(f"Reading markdown from: {DOC_PATH}")
    with open(DOC_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. 预处理 Markdown：将模糊的 <details><summary> 替换为带有章节唯一标识的标题
    # 这解决飞书剥离 HTML 标签后导致多个相同 "4. 图表与数据来源" 混淆的问题
    replacements = [
        ("## 四、LoRA 方法对齐", "## 四、LoRA 方法对齐"),
        ("<summary><strong>4. 图表与数据来源</strong></summary>", "### 【图表及数据】LoRA 方法对齐数据来源", 1), # 第一次出现
        ("<summary><strong>4. 图表与数据来源</strong></summary>", "### 【图表及数据】LoRA dropout=0.10 验证数据来源", 1), # 第二次
        ("<summary><strong>4. 图表与数据来源</strong></summary>", "### 【图表及数据】Ridge 校准数据来源", 1), # 第三次
        ("<summary><strong>4. 图表与数据来源</strong></summary>", "### 【图表及数据】MSE 与 Huber 损失数据来源", 1), # 第四次
    ]

    processed_content = content
    # 清理 details 标签
    processed_content = processed_content.replace("<details>", "").replace("</details>", "")
    
    # 替换 4 个图表来源标题为唯一标题
    processed_content = processed_content.replace("<summary><strong>4. 图表与数据来源</strong></summary>", "### 【图表及数据】数据来源", 1)
    
    # 我们为 12 张图片所在位置直接插入【唯一标志性文本段落】，作为插入锚点，但不删除它们，保证绝对稳定！
    img_anchors = [
        ("01_mpp2_repair_before_after.png", "【位置锚点-01: MPP2修复对比】", "MPP2 修复前后外部指标比较"),
        ("chart2_lora_comparison.png", "【位置锚点-02: LoRA总体指标】", "LoRA 与冻结对照总体指标"),
        ("chart3_patient_slope.png", "【位置锚点-03: 患者变化】", "内部验证患者变化"),
        ("chart4_bootstrap_ci.png", "【位置锚点-04: 重采样区间】", "外部空间重采样区间"),
        ("chart5_pathway_comparison.png", "【位置锚点-05: 30条通路PCC】", "30 条通路 PCC 变化"),
        ("03_lora_dropout_comparison.png", "【位置锚点-06: Dropout对比】", "S0、LoRA dropout=0 与 dropout=0.10 对比"),
        ("04a_ridge_external_metrics.png", "【位置锚点-07: Ridge外部指标】", "Ridge 校准前后外部指标"),
        ("04b_ridge_pathway_delta_r2.png", "【位置锚点-08: Ridge R2变化】", "30 条通路 Raw R² 变化"),
        ("04c_ridge_spatial_interval.png", "【位置锚点-09: Ridge空间区间】", "空间分组下的 Raw R² 变化区间"),
        ("05a_mse_huber_pooled_pcc.png", "【位置锚点-10: Huber PCC对比】", "MSE 与 Huber pooled PCC 对比"),
        ("05b_huber_pathway_direction.png", "【位置锚点-11: Huber方向计数】", "Huber 逐通路改善方向计数"),
        ("05c_mse_huber_mean_pathway_raw_r2.png", "【位置锚点-12: Huber R2比较】", "MSE 与 Huber 平均逐通路 Raw R²"),
    ]

    for img_fn, anchor_text, caption in img_anchors:
        # 在 markdown 中把各个 ![alt](path) 替换为锚点文本 + 图片名
        pattern = r'!\[.*?\]\(.*?' + re.escape(img_fn) + r'\)'
        processed_content = re.sub(pattern, f"\n\n**{anchor_text}**\n\n", processed_content)

    # 2. 写入临时 markdown 文件
    doc_title = "MPP2基线修复与后续实验进展_20260731"
    temp_md_path = "./temp_doc_placeholder.md"
    with open(temp_md_path, "w", encoding="utf-8") as tf:
        tf.write(processed_content)

    print("Creating initial Lark document...")
    create_res = run_lark_cli(["docs", "+create", "--title", doc_title, "--doc-format", "markdown", "--content", "@./temp_doc_placeholder.md"])
    
    if os.path.exists(temp_md_path):
        os.remove(temp_md_path)
        
    if not create_res.get("ok"):
        print("Failed to create document:", create_res)
        return

    doc_info = create_res["data"]["document"]
    doc_id = doc_info["document_id"]
    doc_url = doc_info["url"]
    print(f"Created Doc ID: {doc_id}")
    print(f"Doc URL: {doc_url}")

    # 3. 拉取 with-ids 结构
    fetch_res = run_lark_cli(["docs", "+fetch", "--doc", doc_id, "--detail", "with-ids"])
    if not fetch_res.get("ok"):
        print("Failed to fetch doc structure:", fetch_res)
        return

    xml_content = fetch_res["data"]["document"]["content"]
    print("Searching anchors in xml_content...")
    # 打印前 3 个包含【位置锚点】的 XML 片段
    anchors_found = re.findall(r'<[a-zA-Z0-9_-]+[^>]*id=["\']([^"\']+)["\'][^>]*>.*?【位置锚点.*?</[a-zA-Z0-9_-]+>', xml_content, re.DOTALL)
    print(f"Found {len(anchors_found)} anchor blocks via regex.")
    if len(anchors_found) == 0:
        # 搜寻所有包含【位置锚点】的位置
        pos = xml_content.find("【位置锚点")
        if pos != -1:
            print("XML Snippet around anchor:", xml_content[max(0, pos-100):min(len(xml_content), pos+200)])

    # 4. 逐一匹配唯一锚点段落的 block_id
    for idx, (img_fn, anchor_text, caption) in enumerate(img_anchors):
        # 使用更灵活的正则匹配包含 anchor_text 的节点和 id
        pos = xml_content.find(anchor_text)
        if pos == -1:
            print(f"ERROR: Anchor text not found in XML: {anchor_text}")
            continue
        
        # 向上搜寻最近的 id="xxx"
        sub_xml = xml_content[:pos]
        id_matches = list(re.finditer(r'id=["\']([^"\']+)["\']', sub_xml))
        if not id_matches:
            print(f"ERROR: Could not find id before anchor: {anchor_text}")
            continue

        anchor_block_id = id_matches[-1].group(1)
        print(f"Image {idx+1}/{len(img_anchors)}: Anchor '{anchor_text}' -> Block ID: {anchor_block_id}")

        # 计算本地相对路径
        # 在 markdown 中找原图路径
        img_match = re.search(r'!\[.*?\]\((.*?' + re.escape(img_fn) + r')\)', content)
        if not img_match:
            print(f"Could not resolve original image path for {img_fn}")
            continue
            
        full_img_path = os.path.normpath(os.path.join(DOC_DIR, img_match.group(1)))
        rel_img_path = "./" + os.path.relpath(full_img_path, start=os.getcwd()).replace("\\", "/")

        # 先使用 docs +media-insert 将图片上传插入
        insert_res = run_lark_cli(["docs", "+media-insert", "--doc", doc_id, "--file", rel_img_path, "--align", "center", "--caption", caption])
        if not insert_res.get("ok"):
            print(f"Failed to insert image {img_fn}:", insert_res)
            continue

        inserted_block_id = insert_res["data"]["block_id"]
        print(f"  Uploaded image block: {inserted_block_id}")

        # 使用 block_move_after 移动图片到【对应唯一锚点段落】下方！
        move_res = run_lark_cli(["docs", "+update", "--doc", doc_id, "--command", "block_move_after", "--block-id", anchor_block_id, "--src-block-ids", inserted_block_id])
        if move_res.get("ok"):
            print(f"  SUCCESS: Moved image block {inserted_block_id} to after {anchor_block_id}")
        else:
            print(f"  FAILED to move image block: {move_res}")

    print("\n==========================================")
    print("Precision Import Finished!")
    print(f"Document Access URL: {doc_url}")
    print("==========================================")

if __name__ == "__main__":
    main()
