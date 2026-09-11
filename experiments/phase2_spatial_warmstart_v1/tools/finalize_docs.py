"""同步方案快照及包说明，并完成轻量交付核验；不训练。"""
from pathlib import Path
import ast
import json
import re
import os
from PIL import Image

PKG=Path(__file__).resolve().parents[1]
ROOT=PKG.parents[1]
PLAN=ROOT/'01_指南与解读/部署方案/Phase2完整继承单点模型与空间修正_部署方案_v1_20260910.md'
readme='''# Phase 2完整继承单点模型与空间修正

**状态：方案与包骨架已准备，训练逻辑未实现，尚不可上传后直接训练。**

本轮只有默认种子42，三组：单点续训、固定H/C的纯空间补偿、H/C与空间B联合优化。全部继承阶段一冻结纯回归第5轮H/C；新B为零；不启用LoRA、对比学习或子图采样。

v1.1明确：当前来源是冻结纯回归臂自身最佳第5轮，不是全部阶段一候选的平均PCC冠军；优先运行纯空间补偿。旧图最多8邻居。增加固定β=1预测平滑的本地诊断（不选模），Ridge仅作为默认关闭的备选求解器。第0步一致不能保证训练后或XZY不下降。

- [详细部署方案与流程图](docs/部署方案.md)
- [新对话实现交接](docs/implementation_handoff.md)
- [方案配置](config.json)
- [元数据复制来源](docs/input_sources.json)
- [Gemini润色记录](docs/Gemini润色记录_20260910.json)

只查看计划，可在包内运行：

    python src/main.py --plan

普通入口及run.ps1在未实现状态下退出，不产生伪训练结果。下一段对话完成实现后再提供训练、恢复与外部评价命令。

现有runner.py和run.ps1来自项目模板；src/main.py为占位，inputs带入上一轮必要小型元数据。旧包源码仅供复用参考，尚未复制进src。tools中的两个脚本仅用于生成与核验这次方案，不能当训练入口。

实现完成后，由用户将本目录复制到：

    D:\\AIPatho\\qzs\\code\\phase2_spatial_warmstart_v1

原始输出和权重分别位于runs、weights下本实验的warmstart_no_lora_v1批次目录。模板启动器当前仍待实现batch层级，详见交接清单。常规回传不含权重文件，但包含model_weights.json。

源码、原始结果、权重彼此分离；旧包与旧结果保留。这里没有新实验结果或科研确认状态。
'''
(PKG/'README.md').write_text(readme,encoding='utf-8')
(PKG/'requirements.txt').write_text('''# 规划依赖清单，非已验证运行环境；由实现对话核对已有服务器环境后定稿。
# 不包含LoRA/UNI训练依赖，不自动联网安装。
torch==2.6.0
numpy>=1.24,<3
pandas>=2.1,<3
# 文档工具，仅本地生成/核验说明图时需要：
matplotlib>=3.8
Pillow>=10
''',encoding='utf-8')
(PKG/'tests/README.md').write_text('此目录留给后续实现验证。关键行为见docs/implementation_handoff.md及部署方案第十节。本次没有科研训练实现，未声称训练测试通过。\n',encoding='utf-8')

text=PLAN.read_text(encoding='utf-8')
def rebase(match):
    label,target=match.groups()
    if target.startswith(('https://','http://')):
        return match.group(0)
    resolved=(PLAN.parent/target).resolve()
    relative=os.path.relpath(resolved,PKG/'docs').replace('\\','/')
    return '['+label+']('+relative+')'
snapshot=re.sub(r'\[([^\]]*)\]\(([^)]+)\)',rebase,text)
snapshot='<!-- 本文件为主部署方案的同步快照；生成脚本tools/finalize_docs.py。 -->\n\n'+snapshot
(PKG/'docs/部署方案.md').write_text(snapshot,encoding='utf-8')
config=json.loads((PKG/'config.json').read_text(encoding='utf-8-sig'))
config['parameters']['confirmatory_head_seeds']=[]
config['parameters']['repeat_schedule']='single_seed42_only'
config['plan_version']='v1.1'
config['inputs']['source_selection']={
    'eligible_scope':'no_lora_and_no_contrastive_frozen_regression',
    'within_arm_selection':'max_internal_patient_macro_pathway_pcc_over_stage1_epochs_1_to_5',
    'all_stage1_champion_claim':False,
    'external_metrics_used_for_source_selection':False
}
config['parameters']['execution_order']=['spatial_residual_only','point_continue','spatial_joint']
config['diagnostics']={
    'fixed_prediction_smoothing':{
        'enabled':True,'compute_location':'local_return_analysis','beta':1.0,
        'source':'step0_stage1_predictions','formula':'p + beta * sum_j a_ij * (p_j - p_i)',
        'include_self_weight_in_normalization':True,'used_for_selection':False,
        'external_status':'compute_only_if_fixed_predictions_and_matching_graph_metadata_available'
    },
    'ridge_optional':{
        'enabled':False,'lambda':None,'fit_intercept':False,
        'objective':'sum_squared_residual_error_plus_lambda_times_B_frobenius_squared',
        'status':'optional_solver_requires_prespecified_lambda_protocol_not_a_required_run',
        'guaranteed_external_improvement':False
    }
}
(PKG/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
package=json.loads((PKG/'package.json').read_text(encoding='utf-8'))
package['plan_version']='v1.1'
package['code_version']='v0.1.1-plan'
if not any(x.get('path','').endswith('/inputs') for x in package['code_sources']):
    package['code_sources'].append({'path':'experiments/phase2_softlink_contrastive_v3/inputs','status':'copied_metadata','manifest':'docs/input_sources.json','date':'2026-09-10'})
(PKG/'package.json').write_text(json.dumps(package,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

errors=[];links=0
for p in [PLAN,PKG/'README.md',PKG/'docs/部署方案.md',PKG/'docs/implementation_handoff.md']:
    value=p.read_text(encoding='utf-8')
    for dest in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',value):
        if not dest.startswith(('http://','https://')):
            links+=1
            if not (p.parent/dest).resolve().exists():errors.append('missing link: '+str(p.parent/dest))
    width=None
    for line in value.splitlines():
        if line.startswith('|'):
            n=len(line.split('|'))-2
            if width is None:width=n
            elif n!=width:errors.append('table width: '+str(p))
        else:width=None
for p in PKG.rglob('*.json'):
    try:json.loads(p.read_text(encoding='utf-8-sig'))
    except Exception as exc:errors.append(str(p)+': '+str(exc))
for p in PKG.rglob('*.py'):
    ast.parse(p.read_text(encoding='utf-8-sig'))
assert config['parameters']['primary_seed']==42
assert config['parameters']['confirmatory_head_seeds']==[]
assert len(config['parameters']['arms'])==3
assert config['parameters']['regression_initialization']=='inherit_stage1_C'
assert config['parameters']['hard_max_updates']==2000
assert package['implementation_status']=='planned_not_implemented'
assert '三组均只使用默认种子42' in text
assert '7条训练轨迹' not in text
with Image.open(PKG/'docs/figures/新设计流程.png') as im:
    dimensions=im.size
    im.verify()
metadata=json.loads((PKG/'docs/input_sources.json').read_text(encoding='utf-8'))
result={'status':'passed' if not errors else 'failed','scope':'规划与骨架交付核验，不是训练或模型性能验证',
        'implementation_status':package['implementation_status'],'seed':42,'planned_training_arms':3,
        'source_metadata_files':len(metadata['files']),'local_links_checked':links,
        'json_and_python_syntax':'passed','flowchart_dimensions':dimensions,
        'training_executed':False,'errors':errors}
(PKG/'docs/交付核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
if errors:raise SystemExit(1)
