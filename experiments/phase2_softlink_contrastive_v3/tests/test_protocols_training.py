import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT))
from protocols import build_lopo6_splits, build_original_split, fit_patient_means, fit_zscore
from metrics import mse_decomposition, patient_macro_pathway_pcc, pooled_pcc, regression_metrics, spatial_residual_metrics
from training import PatientBalancedBatchSampler, augmentation_seed, contrastive_lambda, run_stage1, run_stage2, select_best_epoch, select_learning_rate, stage2_task_matrix
from orchestrator import RunKey, build_task_dag, execute_plan, initialise_state, mark_task

def manifest():
    rows=[]
    for p in ['A','B','C','D','E','F']:
        rows += [{'patient':p,'patch_stem':f'{p}t{i}','split':'train'} for i in range(3)]
        rows += [{'patient':p,'patch_stem':f'{p}v{i}','split':'internal_val'} for i in range(2)]
    return pd.DataFrame(rows)

def test_protocol_isolation_and_counts():
    original=build_original_split(manifest()); assert len(original.train_rows)==18 and original.held_out_rows.empty
    folds=build_lopo6_splits(manifest()); assert len(folds)==6
    assert all(len(x.train_rows)==15 and len(x.internal_val_rows)==10 and len(x.held_out_rows)==5 for x in folds)
    assert all(x.fold not in set(x.train_rows.patient)|set(x.internal_val_rows.patient) for x in folds)

def test_lopo_fold_count_contract_8163_933():
    rows=[]
    # Fold A is deliberately sized like the documented original entrypoint.
    counts=[('A',1309,145),('B',1633,187),('C',1633,187),('D',1633,187),('E',1633,187),('F',1631,185)]
    for patient,n_train,n_val in counts:
        rows += [{'patient':patient,'patch_stem':f'{patient}t{i}','split':'train'} for i in range(n_train)]
        rows += [{'patient':patient,'patch_stem':f'{patient}v{i}','split':'internal_val'} for i in range(n_val)]
    fold={x.fold:x for x in build_lopo6_splits(pd.DataFrame(rows))}['A']
    assert (len(fold.train_rows),len(fold.internal_val_rows),len(fold.held_out_rows))==(8163,933,1454)

def test_train_only_fits():
    rows=manifest().iloc[:3]; values=np.array([[0.,0.],[2.,2.],[4.,4.]])
    z=fit_zscore(rows,values); means=fit_patient_means(rows,values)
    assert np.allclose(z.mean,[2,2]) and set(means.patient_means)=={'A'} and len(z.fit_identity)==3

def test_selection_sampling_and_matrix():
    assert select_learning_rate([{'lr':1e-4,'pcc':.5,'zMSE':1},{'lr':1e-5,'pcc':.5,'zMSE':1}])['lr']==1e-5
    assert select_best_epoch([{'epoch':2,'pcc':.4,'zMSE':1},{'epoch':1,'pcc':.4,'zMSE':1}])['epoch']==1
    assert len(stage2_task_matrix())==22
    sampler=PatientBalancedBatchSampler(['A']*4+['B']*4,4); assert all(len(set(x))==len(x) for x in sampler.batches(1))
    assert augmentation_seed('p','f',42,1,2,3)==augmentation_seed('p','f',42,1,2,3)

def test_patient_balanced_sampler_keeps_batch_unique_when_patient_pool_wraps():
    # Two samples per patient and batch=3 forces one patient's permutation to
    # wrap inside the second batch; all four source indices are still feasible.
    batches = PatientBalancedBatchSampler(['A','A','B','B'], 3, seed=42).batches(1)
    assert len(batches) == 2
    assert all(len(batch) == len(set(batch.tolist())) for batch in batches)

def test_double_pcc_decomposition_and_resume(tmp_path):
    target=np.array([[0.],[1.],[0.],[1.]]) ; pred=np.array([[0.],[1.],[1.],[0.]])
    assert patient_macro_pathway_pcc(pred,target,['A','A','B','B'])==0
    assert pooled_pcc(pred,target)==0
    d=mse_decomposition(pred,target,['A','A','B','B']); assert np.isclose(d['mse'],d['demeaned_mse']+d['mean_bias_squared'])
    metrics=regression_metrics(pred,target,['A','A','B','B'],raw_pred=pred*10,raw_target=target*10); assert metrics['rawMAE']==metrics['zMAE']*10
    assert len(metrics['per_patient_pathway'])==2 and 'demeaned_zRMSE' in metrics and 'demeaned_CCC' in metrics
    p=tmp_path/'state.json'; key=RunKey('original','original',42,'r8_regression',5).value(); initialise_state(p,[{'key':key}]); mark_task(p,key,'completed'); assert mark_task(p,key,'completed')['tasks'][key]['status']=='completed'

def test_spatial_residual_metrics_uses_fixed_edges():
    class Graph:
        neighbor_index=np.array([[1],[0]])
        neighbor_weight=np.ones((2,1),dtype=float)
        neighbor_mask=np.ones((2,1),dtype=bool)
    result=spatial_residual_metrics(np.array([[0.],[2.]]),np.array([[0.],[0.]]),Graph())
    assert result['edge_count']==2 and np.isclose(result['weighted_residual_semivariance'],2.0)

def test_dag_reuses_seed42_pilot_and_lopo_has_no_original():
    dag=build_task_dag(manifest(),protocol='original')
    online=[x for x in dag if x['kind']=='online']; pilots=[x for x in dag if x['kind']=='lr_pilot']
    assert len(pilots)==3 and len(online)==17
    assert not any(x['run_key']['seed']==42 and x['run_key']['cell']=='r8_regression' for x in online)
    assert all(x['lr_source_seed']==42 for x in online)
    lopo=build_task_dag(manifest(),protocol='lopo6')
    assert {x['run_key']['protocol'] for x in lopo if 'run_key' in x}=={'lopo6'}
    sensitive=build_task_dag(manifest(),protocol='original',best_epochs={'r2_regression':3})
    known={x['key'] for x in sensitive}
    assert all(dep in known for task in sensitive for dep in task['depends_on'])

def test_callback_training_loop_lambda_selection_and_patience():
    seen=[]
    def step(epoch,update,lam): seen.append((epoch,update,lam)); return {'loss':1}
    stage1=run_stage1(step,lambda epoch:{'pcc':float(epoch),'zMSE':10-epoch},updates_per_epoch=3)
    assert stage1['epochs_completed']==5 and [x[2] for x in seen[:3]]==[0.,.025,.05]
    stage2=run_stage2(lambda *_:None,lambda epoch:{'pcc':1.,'zMSE':1.},max_epochs=60)
    assert stage2['stopped_early'] and stage2['epochs_completed']==25
    assert stage2['selection_start_epoch']==6

def test_stage1_can_capture_only_trainable_state_without_copying_full_model():
    class FullModelMustNotBeCopied:
        def state_dict(self):
            raise AssertionError('full frozen backbone state must not be captured')

    captures=[]
    result=run_stage1(
        lambda *_:{'loss':1.0},
        lambda epoch:{'pcc':float(epoch),'zMSE':10.0-epoch},
        model=FullModelMustNotBeCopied(),
        state_getter=lambda:{'adapter.weight':np.asarray([1.0],dtype=np.float32)},
        checkpoint_callback=lambda kind,epoch,state,row: captures.append((kind,epoch,state)),
    )

    assert result['epochs_completed']==5
    assert captures
    assert all(set(state)=={'adapter.weight'} for _,_,state in captures)

def test_warmup_selection_starts_at_epoch_six_and_saves_each_epoch():
    from training import run_common_warmup
    events=[]
    result=run_common_warmup(
        lambda *_:{},
        lambda epoch:{'pcc':100. if epoch==1 else float(epoch),'zMSE':1.},
        max_epochs=7,
        checkpoint_callback=lambda kind,epoch,state,row: events.append((kind,epoch)),
    )
    assert result['formal']['epoch']==7
    assert ('epoch',1) in events and ('epoch',7) in events
    assert all(epoch>=6 for kind,epoch in events if kind=='formal')

def test_runner_facing_plan_only_writes_raw_plan_and_never_needs_engine(tmp_path):
    config=ROOT/'config.json'
    result=execute_plan(config,tmp_path/'run',tmp_path/'weights','original',(42,),None,False,True)
    assert result['exit_code']==0
    assert (tmp_path/'run'/'raw'/'task_plan.json').is_file()
    assert (tmp_path/'run'/'raw'/'task_state.json').is_file()
    assert execute_plan(config,tmp_path/'run2',tmp_path/'weights2','original',(42,),None,False,False)['exit_code'] != 0
    import runner
    assert runner.main(['--config',str(config),'--run-dir',str(tmp_path/'runner_run'),'--weights-dir',str(tmp_path/'runner_weights'),'--protocol','original','--seeds','42','--plan-only']) == 0
