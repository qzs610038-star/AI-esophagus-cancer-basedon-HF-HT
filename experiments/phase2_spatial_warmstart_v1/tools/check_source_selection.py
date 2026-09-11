"""只核对已有来源记录与方案选择；不拟合新模型、不训练。"""
from pathlib import Path
import csv
import json
import statistics

PKG=Path(__file__).resolve().parents[1]
ROOT=PKG.parents[1]
NEW=ROOT/'experiments/results/phase2_softlink_contrastive_v3/20260909_101623_341_f0de1302'
OLD=ROOT/'experiments/results/phase2_softlink_local_v2/20260908_005325_810_8d306c10'
def read_json(p):return json.loads(p.read_text(encoding='utf-8-sig'))
with (NEW/'analysis/pcc_side_by_side.csv').open(encoding='utf-8-sig',newline='') as f:
    rows=[r for r in csv.DictReader(f) if r['kind'].startswith('stage1_')]
cols=['internal_patient_macro_pathway_pcc','internal_pooled_pcc','external_patient_macro_pathway_pcc','external_pooled_pcc']
ranked=sorted(rows,key=lambda r:-float(r[cols[0]]))
history=read_json(NEW/'raw/original/original/seed_42/stage1/frozen_regression.json')['history']
best=max(history,key=lambda r:r['patient_macro_pathway_pcc'])
assert best['epoch']==5
assert ranked[0]['cell']=='r2_regression' and ranked[0]['stage1_endpoint']=='3'
selected=next(r for r in rows if r['cell']=='frozen_regression')
cfg=read_json(PKG/'config.json')
graph=read_json(OLD/'config_snapshot.json')['graph']
for key in ['radius_in_native_steps','max_neighbors','distance_sigma','image_temperature','self_raw_weight']:
    assert graph[key]==cfg['parameters']['graph'][key],key
with (OLD/'analysis/registration_metrics_by_seed.csv').open(encoding='utf-8-sig',newline='') as f:
    smoothing=[r for r in csv.DictReader(f) if r['arm']=='fixed_smoothing']
assert len(smoothing)==3 and all(r['split']=='internal_val' for r in smoothing)
summary={}
for key in ['pcc','flattened_pcc']:
    values=[float(r[key]) for r in smoothing]
    summary[key]={'mean':statistics.mean(values),'sample_sd':statistics.stdev(values)}
out={
    'status':'passed','date':'2026-09-10','plan_version':'v1.1',
    'source':'frozen_regression_seed42_epoch5',
    'source_selection_scope':'best_within_frozen_regression_formal_epochs_1_to_5_under_no_lora_no_contrastive_constraint',
    'selected_source_is_global_stage1_summary_champion':False,
    'stage1_summary_rankings':[{'cell':r['cell'],'epoch':int(r['stage1_endpoint']),**{c:float(r[c]) for c in cols}} for r in ranked],
    'internal_macro_gap_to_summary_champion':float(ranked[0][cols[0]])-float(selected[cols[0]]),
    'within_source_epoch_history':[{'epoch':r['epoch'],'patient_macro_pathway_pcc':r['patient_macro_pathway_pcc'],'pooled_pcc':r['pooled_pcc']} for r in history],
    'old_graph_parameters_match_new_plan':True,'max_neighbors':graph['max_neighbors'],
    'old_fixed_smoothing':{'beta':1,'registered_split':'internal_val','n_seeds':3,'dual_pcc':summary,'external_record_present':False},
    'scientific_status':'historical_results_registered_pending; no new model fitting',
    'new_training_executed':False
}
(PKG/'docs/来源模型与图参数核验.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':out['status'],'source_best_epoch':best['epoch'],'global_summary_best':'r2_regression_epoch3','internal_macro_gap':out['internal_macro_gap_to_summary_champion'],'old_max_neighbors':graph['max_neighbors'],'old_smoothing':summary,'training':False},ensure_ascii=False,indent=2))
