"""Build a local image gallery and verify the exported file formats."""
from pathlib import Path
import json
import html
from urllib.parse import quote
import xml.etree.ElementTree as ET
from PIL import Image
import fitz

ROOT = Path(__file__).resolve().parents[1]
ITEMS = [
    ('Fig_P2_1_data_flow', '图P2-1 · 数据流程', '开发队列、全视野输入、内部选模与外部评价。'),
    ('Fig_P2_2_model_structure', '图P2-2 · 模型结构', '中心预测与空间残差，以及三臂的配方和训练方式。'),
    ('Fig_P2_3_编码器与空间分支', '图P2-3 · 编码器与空间分支', '双PCC及同种子配对的空间增量。空心标记为均值，误差线为样本SD，浅色点为逐种子值。'),
    ('Fig_P2_4_训练密度消融', '图P2-4 · 训练密度', '等更新次数主分析；点阵仅为采样概念示意。'),
    ('Fig_P2_4S_等50轮敏感性分析', '图P2-4S · 敏感性分析', '同一轨迹前50个完整训练轮次的检查点选择，不增加训练批次。'),
]


def main():
    cards, checks = [], []
    for i, (stem, title, description) in enumerate(ITEMS):
        image_path = ROOT / 'exports' / (stem + '.png')
        with Image.open(image_path) as im:
            assert im.size[0] == 3600
            assert all(abs(x - 300) < .1 for x in im.info['dpi'])
            size, dpi = im.size, im.info['dpi']
        svg = ET.parse(ROOT / 'exports' / (stem + '.svg'))
        texts = len(svg.findall('.//{http://www.w3.org/2000/svg}text'))
        assert texts > 10
        with fitz.open(ROOT / 'exports' / (stem + '.pdf')) as pdf:
            assert len(pdf) == 1
            raster_images = len(pdf[0].get_images(full=True))
            assert raster_images == 0
            assert pdf[0].get_text().strip()
        checks.append(dict(figure=stem, png_size=size, png_dpi=dpi,
                           svg_editable_text_nodes=texts, pdf_pages=1, pdf_raster_images=raster_images))
        url = lambda ext: quote('exports/' + stem + '.' + ext, safe='/')
        cards.append(f'''<section id="f{i+1}"><div class="heading"><h2>{html.escape(title)}</h2>
<div class="links"><a href="{url('png')}">PNG 高清</a><a href="{url('svg')}">SVG 可编辑</a><a href="{url('pdf')}">PDF 矢量</a></div></div>
<p>{html.escape(description)}</p><a href="{url('png')}"><img src="{url('png')}" alt="{html.escape(title)}" loading="lazy"></a></section>''')
    nav = ''.join(f'<a href="#f{i+1}">{html.escape(t)}</a>' for i, (_, t, _) in enumerate(ITEMS))
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phase 2 首批图片 · v1</title><style>
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f4f6f8;color:#243247;font:16px/1.6 'Microsoft YaHei',sans-serif}main{max-width:1240px;margin:auto;padding:36px 28px 60px}header{margin-bottom:24px}h1{font-size:30px;line-height:1.3;margin:0 0 12px}header p{color:#64748b}nav{display:flex;gap:10px;flex-wrap:wrap}a{color:#256185;text-decoration:none}a:hover{text-decoration:underline}nav a{padding:6px 12px;background:white;border:1px solid #dce3eb;border-radius:5px;font-size:14px}section{background:white;padding:25px 25px 16px;margin:22px 0;border:1px solid #e2e7ed;border-radius:8px;scroll-margin-top:18px}h2{font-size:22px;margin:0}.heading{display:flex;justify-content:space-between;align-items:center;gap:18px;flex-wrap:wrap}.links{display:flex;gap:16px;font-size:14px}section p{color:#64748b;margin:8px 0 20px}img{display:block;width:100%;height:auto}footer{color:#64748b;font-size:14px}@media(max-width:640px){main{padding:24px 12px}section{padding:18px 10px}h1{font-size:24px}h2{font-size:19px}}
</style><main><header><h1>Phase 2 首批图片 · 中文审阅版</h1><p>4张主图 + 1张补充图 · 2026-09-22 · 点击图片查看高清原图。各图均保留可编辑矢量版本。</p><nav>''' + nav + '</nav></header>' + ''.join(cards) + '''<footer>来源、完整图注与核对说明见同目录README及首批图片图注。数据来自已接纳主实验与密度消融；图中SD表示三种子的样本标准差。</footer></main></html>'''
    (ROOT / '图集.html').write_text(page, encoding='utf-8')
    (ROOT / 'export_checks.json').write_text(json.dumps({'status': 'pass', 'checks': checks}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Gallery ready; {len(checks)} figures verified in PNG / editable-text SVG / vector PDF.')


if __name__ == '__main__':
    main()
