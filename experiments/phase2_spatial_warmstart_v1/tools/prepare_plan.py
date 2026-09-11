"""创建一次性方案骨架和说明图；不实现或执行科研训练。"""
from pathlib import Path
import json
import shutil

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parents[1]
EXPERIMENT = PKG.name
for folder in ('docs/figures', 'src', 'tests', 'inputs'):
    (PKG / folder).mkdir(parents=True, exist_ok=True)

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

if not (PKG / 'package.json').exists():
    # 只复制模板正式文件，不复制其测试产物、缓存或旧运行目录。
    for name in ('runner.py', 'run.ps1', 'config.json', 'package.json', 'README.md', 'requirements.txt'):
        destination = PKG / name
        if not destination.exists():
            shutil.copy2(ROOT / 'experiments/_template' / name, destination)
    launcher = (PKG / 'run.ps1').read_text(encoding='utf-8-sig')
    guard = """$planningPackage = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'package.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($planningPackage.implementation_status -ne 'implemented') {
    throw 'PLAN ONLY: training is not implemented. Read docs/implementation_handoff.md.'
}
"""
    launcher = launcher.replace("$ErrorActionPreference = 'Stop'", "$ErrorActionPreference = 'Stop'\n" + guard)
    (PKG / 'run.ps1').write_text(launcher, encoding='utf-8-sig')
    template_config = json.loads((ROOT / 'experiments/_template/config.json').read_text(encoding='utf-8'))
    runid = '20260909_101623_341_f0de1302'
    oldrun = ROOT / 'experiments/results/phase2_softlink_contrastive_v3' / runid
    source_cache = json.loads((oldrun / 'raw/original/original/seed_42/cache/frozen_regression/e5.json').read_text(encoding='utf-8'))
    weights = json.loads((oldrun / 'model_weights.json').read_text(encoding='utf-8'))
    source_weight = next(e for e in weights['entries'] if e['cell'] == 'frozen_regression' and e['endpoint'] == 5)
    endpoint = source_weight['weight_directory'] + '\\epoch_5.pt'
    config = {
        'experiment_id': EXPERIMENT, 'plan_version': 'v1.1', 'implementation_status': 'planned_not_implemented',
        'batch_id': 'warmstart_no_lora_v1', **template_config,
        'inputs': {
            'source_experiment': 'phase2_softlink_contrastive_v3', 'source_run_id': runid,
            'source_cell': 'frozen_regression', 'source_seed': 42, 'source_endpoint': 5,
            'stage1_checkpoint': endpoint, 'feature_cache': source_cache['cache'],
            'graph_feature_cache': source_cache['graph_cache'],
            'source_paths_status': 'recorded_server_paths_require_preflight',
            'small_inputs': 'inputs', 'xzy_source': 'resolve_from_source_manifest_at_implementation'
        },
        'parameters': {
            'arms': [
                {'id': 'point_continue', 'train_H': True, 'train_C': True, 'train_B': False, 'lr_HC': 3e-5, 'lr_B': None},
                {'id': 'spatial_residual_only', 'train_H': False, 'train_C': False, 'train_B': True, 'lr_HC': None, 'lr_B': 3e-4},
                {'id': 'spatial_joint', 'train_H': True, 'train_C': True, 'train_B': True, 'lr_HC': 3e-5, 'lr_B': 3e-4}
            ],
            'source_checkpoint_shared_across_all_arms_and_seeds': True,
            'primary_seed': 42, 'confirmatory_head_seeds': [],
            'repeat_schedule': 'single_seed42_only', 'residual_only_repeats': 1,
            'seed_scope': 'stage2_conditional_on_single_seed42_stage1_checkpoint',
            'backbone_trainable': False, 'lora_enabled': False, 'contrastive_enabled': False,
            'input_dim': 1536, 'hidden_dim': 256, 'output_dim': 30, 'dropout': 0.3,
            'regression_initialization': 'inherit_stage1_C', 'shared_initialization': 'inherit_stage1_H',
            'spatial_initialization': 'zeros', 'baseline_step0_selectable': True,
            'batch_mode': 'full_batch', 'loss': 'all_point_pathway_mean_zMSE',
            'optimizer': 'AdamW', 'weight_decay': 1e-4, 'lr_schedule': 'constant',
            'initial_update_budget': 1000, 'hard_max_updates': 2000,
            'extend_at_1000_if_not_early_stopped': True,
            'validation_every_updates': 10, 'min_updates_before_early_stop': 200,
            'patience_validation_checks': 20, 'min_delta_patient_macro_pcc': 1e-4,
            'selection_order': ['max_patient_macro_pathway_pcc', 'min_zMSE', 'min_update'],
            'graph': {'radius_in_native_steps': 1.5, 'max_neighbors': 8, 'distance_sigma': 1.0,
                      'image_temperature': 0.2, 'self_raw_weight': 1.0,
                      'scope': 'within_patient_and_partition', 'use_labels': False},
            'evaluation_precision': 'float32', 'initial_prediction_max_abs_tolerance': 1e-6,
            'external_selection_allowed': False
        }
    }
    write_json(PKG / 'config.json', config)
    write_json(PKG / 'package.json', {
        'experiment_id': EXPERIMENT, 'code_version': 'v0.1.1-plan', 'plan_version': 'v1.1',
        'entrypoint': 'src/main.py', 'args': [], 'demo_only': True,
        'implementation_status': 'planned_not_implemented',
        'code_sources': [
            {'path': 'experiments/_template', 'copied_files': ['runner.py','run.ps1','config.json','package.json','README.md','requirements.txt'],
             'date': '2026-09-10', 'note': '模板启动器与目录合同；配置说明已定制，run.ps1增加未实现状态提示'},
            {'path': 'experiments/phase2_softlink_contrastive_v3', 'status': 'reference_only_not_copied', 'note': '后续复制所需数据、模型、图、指标工具到本包，禁止上级目录运行时导入'},
            {'path': 'experiments/phase2_softlink_local_v2', 'status': 'reference_only_not_copied', 'note': '旧空间结构与成对训练参考'}
        ]
    })
    # 仅带入独立运行必需的小型元数据，保留相对目录。
    input_source = ROOT / 'experiments/phase2_softlink_contrastive_v3/inputs'
    copied = []
    for p in input_source.rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.json', '.csv', '.yaml', '.yml'} and p.stat().st_size < 5_000_000:
            rel = p.relative_to(input_source)
            dest = PKG / 'inputs' / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            copied.append({'source': str(p.relative_to(ROOT)), 'destination': 'inputs/' + rel.as_posix(), 'bytes': p.stat().st_size})
    write_json(PKG / 'docs/input_sources.json', {'date': '2026-09-10', 'status': 'copied_metadata_not_live_server_verified', 'files': copied})

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_xlim(0, 15); ax.set_ylim(-.15, 10); ax.axis('off')
def box(x, y, w, h, text, color='#eef3f8', size=12, bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',linewidth=1.2,edgecolor='#8b9eae',facecolor=color))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color='#243b50',fontweight='bold' if bold else 'normal')
def arrow(x,y,xx,yy):
    ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'-|>','color':'#627c90','lw':1.6})
ax.text(.3,9.65,'新设计：保留已经会预测的单点模型，再学习空间修正',fontsize=20,weight='bold',color='#243b50')
box(.3,8.2,3.7,1.05,'已有冻结 UNI2-h 图像特征\n1536 维；不再训练骨干')
box(4.7,8.2,4.2,1.05,'完整加载阶段一第 5 轮 H＋C\nH：1536→256；C：256→30','#e4f1ed',13,True)
box(9.6,8.2,5.0,1.05,'建立同一单点起点\n空间 B＝0 时，评估预测与原模型一致','#e4f1ed')
arrow(4,8.72,4.62,8.72); arrow(8.9,8.72,9.52,8.72)
box(.3,6.55,14.3,.8,'固定空间邻域：同一患者、同一数据划分内，用坐标＋冻结图像特征找邻居；不用通路标签','#f7f1df',13)
arrow(7.5,8.15,7.5,7.43)
for x in (2.5,7.5,12.5): arrow(x,6.5,x,5.95)
box(.3,4.1,4.4,1.75,'单点续训对照\n保留 H、C 并小学习率继续训练\n不加空间 B\n检验：多训练本身带来多少变化','#eef3f8')
box(5.3,4.1,4.4,1.75,'优先：纯空间补偿\n固定 H、C，只训练新 B\n原单点预测＋B×邻域表征差\n检验：空间信息单独能补多少','#e4f1ed')
box(10.3,4.1,4.4,1.75,'空间联合优化\n小学习率续训 H、C，同时训练 B\n原单点预测与空间修正共同调整\n检验：共同调整是否进一步改善','#f7e9df')
for x in (2.5,7.5,12.5): arrow(x,4.02,x,3.55)
box(.3,2.6,14.3,.85,'全量训练：每步一次参数更新；1000 步检查预算，未早停则继续至最多 2000 步\n每 10 步内部验证，至少 200 步后允许早停；达到上限仍改善，标注尚未收敛','#eef3f8',12)
arrow(7.5,2.52,7.5,2.05)
box(.3,1.12,14.3,.8,'按内部逐患者逐通路平均 PCC 选择最佳（包含第 0 步）；固定模型后再评估 XZY\n同时展示逐患者逐通路平均 PCC 与整体展平 pooled PCC；不以 XZY 调参','#e4f1ed',12)
box(.3,.05,14.3,.65,'与此前阶段二的关键区别：过去只继承 H、重置 C、最多 60 次更新 → 现在继承 H＋C、B 从零开始、检查充分训练','#fff6e7',11)
fig.tight_layout(pad=.4)
for suffix in ('png','svg'):
    fig.savefig(PKG / 'docs/figures' / ('新设计流程.'+suffix),dpi=170,bbox_inches='tight',facecolor='white')
plt.close(fig)
print(json.dumps({'package':str(PKG),'status':'planned_not_implemented','training_executed':False},ensure_ascii=False))
