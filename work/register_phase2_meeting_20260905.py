import copy
import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

root = Path('D:/AI空间转录病理研究/PFMval_new')
now = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
directive_id = 'DIR-20260905-001'
note_path = 'project_state/governance/Phase2会议决策_20260905.md'
contract_id = 'phase2_all_metrics_retained_v3_20260905'

def read_json(path):
    return json.loads((root / path).read_text(encoding='utf-8-sig'))

def write_json(path, value):
    (root / path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

state = read_json('project_state/current_state.json')
registry = read_json('experiments/experiment_registry.json')
original_experiments = copy.deepcopy(registry['experiments'])
docs = read_json('project_state/document_registry.json')
revision = state['state_revision'] + 1
assert directive_id not in state['active_directive_ids']
assert not (root / note_path).exists()

summary = 'Phase 2选定软对比联合学习作为当前研究主线，优先针对本数据调优，使其相对冻结UNI2-h加两层MLP的对照获得更清楚的指标改善；具体模型改动后续讨论。全部已列指标暂时待定，后续实验保留计算和记录，逐通路平均PCC与展平/整体PCC均保留，论文采用及主副线后续再定。指标主要用于本项目内部前后与实验对照，暂不开展跨论文实验数值优劣比较。六折留一患者继续暂缓；基因重建、稠密/稀疏验证、多基础模型和Phase 2/3衔接补充实验后置，在软对比指标改善后再推进。本次只登记决策，不启动训练或改变既有结果确认状态。'
metric_names = [
    'mean_per_pathway_pcc', 'pooled_pcc', 'mean_per_pathway_z_rmse',
    'mean_per_pathway_ccc', 'mean_per_pathway_spearman', 'mean_residual_morans_i',
    'mean_per_pathway_masked_ssim', 'mean_per_pathway_raw_r2',
    'mean_per_pathway_z_mae', 'mean_per_pathway_raw_mae', 'pooled_z_mse',
    'train_sd_normalized_nrmse', 'mean_spot_topk_pathway_overlap',
]
publication_policy = {
    'status': 'all_metrics_retained_publication_selection_pending',
    'decision_id': directive_id,
    'paper_primary': [],
    'paper_supplementary': [],
    'retained_metrics': metric_names,
    'retain_distribution_summaries_and_95ci_metadata': True,
    'comparison_scope': 'within_project_before_after_and_matched_experiments',
    'cross_paper_numeric_ranking': 'deferred_by_user',
    'literature_method_reference': 'retained',
    'user_publication_preference': '后续根据本项目及对比验证实验，选择有优势的指标发表；两种PCC暂时都保留，后续看哪个较高再选择。',
    'pcc_name_mapping': {
        '逐通路平均PCC': 'mean_per_pathway_pcc',
        '用户所称平均池化PCC': 'pooled_pcc（目前已有的展平/整体PCC，不是先做图像平均池化）',
    },
    'methodological_warning': '两种PCC的统计对象与汇总不同，不能因某一种绝对值更高就认定模型更好。发表选择如在观察结果后形成，应说明其探索性并保留完整指标，不能隐藏不利结果或冒充预先指定主终点；文献定义不清也不能使两种公式等价。此为方法说明，不改变用户的暂缓裁决。',
    'model_selection_boundary': '监督预处理、模型改动、超参数及checkpoint选择仅使用训练或内部验证数据，不使用外部测试成绩调参。',
    'applicability': '全部保留不等于强行填值；SSIM无有效窗口、Top-k未预定义、单患者泛化CI不可估计等情况保留明确状态。',
}

program = registry['phase2_supplemental_program']
old_contract = copy.deepcopy(program['metric_registration_contract'])
program.setdefault('metric_registration_contract_history', []).append(old_contract)
contract = copy.deepcopy(old_contract)
contract.update(id=contract_id, effective_at=now, status='active',
                scope='当前软对比主线、冻结直接回归对照及后续Phase 2实验；最终发表指标待定',
                supersedes_publication_roles_of=old_contract['id'])
contract['metric_roles'] = {'paper_primary': [], 'paper_supplementary': [], 'retained_candidates': metric_names}
contract['publication_selection_policy'] = publication_policy
contract['aggregation_protocol']['pooled_pcc'] = '在每个评价集合将点位×通路矩阵展平后计算PCC；与患者优先逐通路平均结果分开记录'
program['metric_registration_contract'] = contract
program.update(status='soft_contrastive_improvement_first', updated_at=now,
               decision_id=directive_id, current_method='软对比联合学习（现有残差纠错方案作为改进起点）',
               method_reference_experiment_id='mpp2_cpgcr_probe_v001_20260811',
               method_reference_arm='CPGCR', baseline_role='冻结UNI2-h加两层MLP作为固定对照；不是当前选定研究主线',
               supplementary_execution_status='deferred_until_soft_contrastive_improvement_review',
               deferred_designs=['six_fold_leave_one_patient_out_deferred_by_user_20260905'],
               priority_order=['先诊断并针对本数据改进软对比联合学习', '指标改善后再推进基因重建、密度、多模型及Phase 2/3衔接补充实验'],
               training_authorized_by_this_decision=False)
program['source_documents'].append(note_path)
registry['current_mpp_policy'].update(
    effective_date='2026-09-05', decision=summary,
    rationale='用户本轮会议选择软对比学习继续作为主线；已有成绩尚未证实相对简单回归有稳定优势，提升是后续目标。',
    next_recommended_experiment='优先讨论针对本数据的软对比改进方案；六折暂缓，其余补充实验后置；本轮不执行实验。',
    reference_doc=note_path, directive_id=directive_id)
registry['updated_at'] = now
planned_ids = program['planned_experiment_ids']
for exp in registry['experiments']:
    if exp['id'] not in planned_ids:
        continue
    exp['priority'] = 'P2'
    exp['execution_schedule'] = 'deferred_until_soft_contrastive_improvement_review'
    exp['schedule_decision_id'] = directive_id
    exp['next_action'] = 'deferred_soft_contrastive_improvement_first_then_revisit_protocol'
    exp['metric_contract_id'] = contract_id
    exp['planned_metric_registration']['paper_primary'] = []
    exp['planned_metric_registration']['paper_supplementary'] = []
    exp['planned_metric_registration']['retained_candidates'] = metric_names
    exp['planned_metric_registration']['publication_selection_status'] = 'pending'
    exp['planned_metric_registration'].pop('internal_diagnostic', None)
    if 'excluded_current_design' in exp:
        exp['excluded_current_design'] = program['deferred_designs'][0]

state['state_revision'] = revision
state['updated_at'] = now
state['active_directive_ids'] = [x for x in state['active_directive_ids'] if x != 'DIR-20260904-001'] + [directive_id]
phase2 = state['phase2_program']
phase2.update({key: copy.deepcopy(program[key]) for key in (
    'status', 'updated_at', 'decision_id', 'current_method', 'method_reference_experiment_id',
    'method_reference_arm', 'baseline_role', 'supplementary_execution_status',
    'deferred_designs', 'priority_order', 'training_authorized_by_this_decision')})
phase2['publication_metric_policy'] = publication_policy
phase2['decision_record'] = note_path
phase2['pending_inputs'] = ['targeted_soft_contrastive_model_revision', 'improvement_criterion_and_internal_selection_protocol', 'final_publication_metric_selection', 'final_cohort_and_input_contracts']
phase2['source_documents'].append(note_path)
state['superseded_conclusions'].extend([
    '2026-09-05会议前建议将直接通路回归作为默认研究主线；当前用户选择软对比联合学习，直接回归保留对照。',
    '既有指标合同中的发表主副线为历史登记口径；当前全部指标保留、最终发表选择待定。',
    '2026-09-04补充实验优先准备排序被2026-09-05软对比改进优先取代；原计划与已有结果继续保留。',
])

note = f'''# Phase 2 会议决策记录

会议日期：2026年9月5日。登记时间：{now}。

性质：用户明确决定的当前研究方向与工作排序；不是新增实验结果或性能接纳。

## 当前决定

1. **全部指标暂时待定，后续实验全部保留。** 既有主线、副线分工暂不作为最终发表方案。保留逐通路平均PCC、展平/整体PCC、标准化均方根误差、一致性相关系数、排序相关系数、空间残差自相关、掩膜结构相似性、决定系数、标准化及原尺度绝对误差、整体标准化均方误差、训练标准差归一化误差，以及条件适用的前k通路重合率；保留通路分布和置信区间信息。不可计算的项目如实记录原因，不填零或虚构数值。
2. **对比以本项目内部为主。** 主要用于模型改进前后、不同方案及验证实验的同口径对比；暂时放弃与其他论文跨数据集、跨预测任务的实验数值优劣比较。论文仍可提供方法参考。
3. **两种PCC均保留，最后再决定发表方式。** 用户所称“平均池化PCC”在当前代码中对应展平/整体PCC；此处没有新增计算方法，也没有将它改成图像特征平均池化。
4. **Phase 2仍选择软对比联合学习作为当前基础方案和改进主线。** 现有软关系残差纠错实现作为起点；冻结UNI2-h加两层MLP保留为最简单的直接回归对照。当前尚未观察到软对比相对对照的稳定明显提升，不能写成已证实的统计显著增益。
5. **第一优先级是有针对性的调优。** 先诊断现有方法与本数据的不匹配，争取PCC或其他指标相对对照取得更清楚、可验证的改善；具体模型改动、训练方案和提升标准后续讨论。本次不把多篇论文方法的组合本身当成有效性证据。
6. **六折实验继续暂缓；补充实验后置。** 当前不安排六折留一患者。先改进软对比；指标改善后，再推进基因预测后重建通路、稠密/稀疏验证、更换基础模型，以及Phase 2与Phase 3衔接相关补充实验。原计划仍保留，不标记为取消或完成。

## 发表指标意向与方法说明

用户原意记录：后续根据本项目及对比验证实验“找出有优势的指标进行发表”；两种PCC先都保留，“哪个PCC指标比较高，我们就选用哪个指标”。最终发表指标及主副线尚未裁定。

方法说明（与上述用户意向分开记录）：逐通路平均PCC与展平PCC的统计对象不同，不能用两种公式之间的绝对大小证明模型改善；应在每一种固定定义内比较候选与对照。若发表选择在观察结果后形成，应说明其探索性、保留完整指标记录并披露定义，不隐藏不利结果或将事后选择冒充预先指定的主要指标。文献未明确PCC算法的观察未在本次登记中作为全面文献调查结论确认。此说明不是新增执行门槛。

监督预处理、模型设计、超参数及模型保存点的选择仍仅使用训练集或内部验证集，不利用外部测试成绩反向调优。暂缓六折不等于已获得跨患者泛化证据。“明显提升”是研究目标，提升幅度与统计方法尚未确定。

## 对既有记录的影响

- 取代会议前报告中“建议直接回归作为默认研究主线”的建议；该报告保留为会前分析，不改写历史内容。
- 旧指标合同和已确认结果原样保留，用于追溯计算定义与既有成绩；新合同将发表角色改为待定，后续实验保留所有指标。
- 取代“等本次组会再决定是否做六折”的待议状态，现为用户明确继续暂缓。
- 取代补充实验优先推进的排序；三项已登记补充实验仍未运行，优先级后置。
- 本次仅授权本地会议决策及相关状态登记；未启动训练、修改模型、写入服务器、发布或接纳新结果。团队工作计划原文不改写。

## 查阅入口

- [当前机器状态](../current_state.json)
- [用户决策事件记录](../directives.jsonl)
- [实验与指标登记](../../experiments/experiment_registry.json)
- [会前整体分析报告](../../01_指南与解读/分析报告/Phase2整体分析报告_模型指标与补充实验_20260905.md)
'''

changed = ['project_state/current_state.json', 'project_state/directives.jsonl', 'experiments/experiment_registry.json', 'project_state/document_registry.json', 'CURRENT_STATE.md', 'PROJECT_GUIDE.md', 'experiments/experiment_progress.md', 'experiments/experiment_dashboard.md', note_path]
directive = dict(event_type='directive', directive_id=directive_id, issued_at=now,
                 summary=summary, scope='phase2,project_maintenance,paper_output',
                 topic='phase2_soft_contrastive_priority_and_metric_selection_meeting', status='active',
                 supersedes=['DIR-20260904-001'], amends=['DIR-20260904-005'],
                 effective_from_revision=revision, affected_files=changed,
                 source='explicit_user_instruction', decision_record=note_path,
                 publication_metric_policy=publication_policy)
event = dict(event_type='status_update', directive_id='DIR-20260904-001', status='superseded', changed_at=now, superseded_by=directive_id)

docs['documents'].append(dict(doc_id='phase2-meeting-decision-20260905', path=note_path,
    category='会议决策', scope='phase2', authority='derived', lifecycle='active',
    verified_at=now, state_revision=revision, supersedes=[], superseded_by=[],
    truth_sources=['project_state/directives.jsonl', 'project_state/current_state.json'],
    purpose='登记软对比优先、全部指标待定保留、项目内部对照及六折与补充实验暂缓的用户决定',
    created='2026-09-05', tags=['Phase2', '会议决策'], connectivity_modes=['offline'], availability='local_only'))
docs['updated_at'] = now
docs['state_revision'] = revision
for key in ('total', 'live', 'active'):
    docs['summary'][key] += 1

meeting_block = f'''## 当前 Phase 2 会议决策（2026-09-05）

- 当前主线：软对比联合学习，优先针对本数据改进；冻结UNI2-h加两层MLP保留为对照。
- 全部指标暂时待定且继续保留；逐通路平均PCC与展平/整体PCC均记录，最终发表指标后续再定。
- 比较范围：本项目内部前后及方案对照；暂不进行跨论文实验数值优劣比较。
- 排序：先改进软对比指标；六折留一患者暂缓；基因重建、密度、多基础模型及Phase 2/3衔接补充实验后置。
- 用户倾向后续选取有优势的指标发表；两种PCC的绝对高低不能互证优越，需保留完整结果和明确计算定义。
- 本轮只登记决定，未训练、未产生或接纳新结果。详细记录：[{note_path}]({note_path})。

'''

# 用户已授权的范围内直接维护派生视图；保留所有历史结果表。
views = {}
for path in ('CURRENT_STATE.md', 'PROJECT_GUIDE.md', 'experiments/experiment_progress.md', 'experiments/experiment_dashboard.md'):
    text = (root / path).read_text(encoding='utf-8-sig')
    text = re.sub(r'^> source_sha256:.*\n', '> 本次按会议决策更新相关导航；不生成普通文档哈希。\n', text, flags=re.M)
    if path == 'CURRENT_STATE.md':
        text = re.sub(r'^> AUTO-GENERATED.*$', '> 按机器状态和用户决策维护；本次为会议决策的范围内更新。', text, flags=re.M)
        text = re.sub(r'^> State revision:.*$', f'> State revision: `{revision}` | Updated: `{now}`', text, flags=re.M)
        text = re.sub(r'^- `DIR-20260904-001`.*\n', '', text, flags=re.M)
        text = text.replace('## Current directives\n', f'## Current directives\n\n- `{directive_id}` [phase2/meeting_decision]: {summary}\n')
        text = text.replace('- `DIR-20260904-005` [phase2/supplemental_experiment_preparation_v1]:', '- `DIR-20260904-005` [phase2/supplemental_experiment_preparation_v1]（补充计划仍保留；优先级由9月5日会议修订）:')
        text = re.sub(r'## Phase 2 supplemental program\n.*?(?=\n## |\Z)', meeting_block.rstrip(), text, flags=re.S)
    else:
        if path.endswith('experiment_progress.md'):
            text = text.replace('> 此文件由 `experiments/experiment_registry.json` 自动生成，禁止手工维护。', '> 依据实验登记生成；本次按用户会议决定更新相关进度。')
            text = re.sub(r'^> state_revision:.*$', f'> state_revision: `{revision}`；updated_at: `{now}`', text, flags=re.M)
        if path.endswith('experiment_dashboard.md'):
            text = re.sub(r'^> Auto-generated.*$', f'> 依据实验登记维护；会议决策更新：{now}。', text, flags=re.M)
            text = re.sub(r'^> ✅ \*\*当前 MPP 主线.*$', '> ✅ **当前主线（2026-09-05）**：软对比联合学习；冻结UNI2-h加两层MLP保留为固定对照。', text, flags=re.M)
            text = re.sub(r'^> 下一步推荐：.*$', '> 下一步：优先针对本数据讨论软对比改进，指标提升后再开展后续补充；六折继续暂缓。', text, flags=re.M)
        block = meeting_block
        if path.startswith('experiments/'):
            block = block.replace(f']({note_path})', f'](../{note_path})')
        split = text.find('\n## ')
        text = text[:split] + '\n' + block + text[split:] if split >= 0 else text + '\n' + block
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith('|'):
            continue
        for previous in original_experiments:
            if previous['id'] in planned_ids and previous['id'] in line:
                line = line.replace(previous['next_action'], '暂缓：先改进软对比，再补协议与实验')
                if path.endswith('experiment_dashboard.md') and line.startswith('| P0 |'):
                    line = line.replace('| P0 |', '| P2 |', 1)
        lines[i] = line
    views[path] = '\n'.join(lines) + '\n'

write_json('project_state/current_state.json', state)
write_json('experiments/experiment_registry.json', registry)
write_json('project_state/document_registry.json', docs)
with (root / 'project_state/directives.jsonl').open('a', encoding='utf-8') as stream:
    stream.write(json.dumps(event, ensure_ascii=False) + '\n')
    stream.write(json.dumps(directive, ensure_ascii=False) + '\n')
(root / note_path).write_text(note, encoding='utf-8')
for path, value in views.items():
    (root / path).write_text(value, encoding='utf-8')

# 最小核验：保存可读、决策引用一致、已存在实验结果未被改写。
saved = read_json('experiments/experiment_registry.json')
for before, after in zip(original_experiments, saved['experiments']):
    if before['id'] not in planned_ids:
        assert before == after, before['id']
    else:
        assert after['status'] == 'planned' and after['metric_registration_status'] == 'pending_not_run'
assert saved['phase2_supplemental_program']['metric_registration_contract_history'][-1] == old_contract
assert read_json('project_state/current_state.json')['phase2_program']['decision_id'] == directive_id
assert read_json('project_state/document_registry.json')['documents'][-1]['path'] == note_path
events = [json.loads(line) for line in (root / 'project_state/directives.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
assert events[-1]['directive_id'] == directive_id
assert all((root / path).exists() for path in changed)
print(json.dumps({'revision': revision, 'updated_files': changed, 'existing_experiment_results_unchanged': True, 'planned_experiments_still_not_run': True}, ensure_ascii=False, indent=2))
