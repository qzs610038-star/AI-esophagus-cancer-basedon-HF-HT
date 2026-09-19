import copy
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import point_table_from_cache_arrays
from graph import build_split_graph, gather_neighbors
from model import build_model
from search import build_extension_plan, build_point_search_plan, build_spatial_search_plan, record_trial_result, replication_queue, search_plan_from_dict
from train import build_lr_scheduler, build_optimizer, prepare_training_state, train_arm

def cfg():
    return {"data":{"input_dim":4,"output_dim":2},"model":{"hidden_dim":3,"dropout":.2,"shared_bias":True,"readout_bias":True,"spatial_bias":False},
            "graph":{"hops":1,"directed":True,"radius_in_native_steps":1.5,"max_neighbors":4,"distance_sigma":1.,"use_image_similarity":True,"image_temperature":.2,"self_raw_weight":1.},
            "training":{"optimizer":"AdamW","learning_rate":1e-3,"weight_decay":1e-4,"b_lr_multiplier":3.,"lr_schedule":"warmup_cosine","warmup_epochs":1,"cosine_min_lr_ratio":.01,"batch_size":2,"keep_last_batch":True,"max_epochs":2,"precision":"float32","amp":False,"tf32":False},
            "selection":{"formal_start_epoch":1,"early_stop_count_start_epoch":2,"early_stop_patience":9,"early_stop_min_delta":1e-4,"checkpoint_tolerance":1e-6},
            "randomness":{"model_seed_offset":0,"center_seed_offset":100,"dropout_seed_offset":200},
            "search":{"sampler":"fallback","optuna_version":"5.0.0","point_sampler_seed":9,"spatial_sampler_seed":10,"initial_trials":24,"anchor_trials":2,"random_trials":6,"tpe_trials":16,"top_k":3,"replication_seeds":[42,43,44],"spatial_restricted_trials":18,"spatial_joint_trials":6,"extension_trials":12,"extension_max_per_stage":1,"extension_edge_trigger":True,"extension_last_six_gain":.002,"optimizer_choices":["AdamW","Adam"],"learning_rate_log_range":[1e-5,1e-3],"adamw_weight_decay_log_range":[1e-6,1e-2],"batch_size_choices":[32,64,128,256],"hidden_dim_choices":[128,256,512],"dropout_choices":[0,.1,.2,.3,.5],"lr_schedule_choices":["constant","warmup_cosine"],"graph_radius_neighbor_choices":[[1.,4],[1.5,8],[2.,12],[2.5,16]],"graph_distance_sigma_choices":[.5,1.,2.],"graph_image_temperature_log_range":[.05,.8],"graph_self_raw_weight_choices":[.5,1.,2.],"b_lr_multiplier_choices":[.3,1.,3.]}}

def table():
    rng=np.random.default_rng(3); n=8
    return point_table_from_cache_arrays(features=rng.normal(size=(n,4)).astype("float32"), labels_z=rng.normal(size=(n,2)).astype("float32"), patient_ids=["a"]*4+["b"]*4, slide_ids=["s1"]*4+["s2"]*4, spot_ids=[f"p{i}" for i in range(n)], splits=["train"]*4+["internal_val"]*4, x=[0,1,0,1,0,1,0,1],y=[0,0,1,1,0,0,1,1],pathway_names=["x","y"])

def geometry():
    return pd.DataFrame({"patient_id":["a","b"],"slide_id":["s1","s2"],"s":[1.,1.],"coordinate_unit":["grid","grid"],"patch_coverage_size":[1.,1.],"source":["test","test"],"status":["verified","verified"]})

def test_hcb_is_dynamic_zero_and_graph_uses_boundary():
    model=build_model(cfg(),"spatial")
    assert model.shared.weight.shape == (3,4) and model.spatial_head.weight.shape == (2,3)
    assert torch.count_nonzero(model.spatial_head.weight) == 0
    t=table(); graph=build_split_graph(t,"train",cfg(),geometry_table=geometry()); got=gather_neighbors(graph,[t.identities[0]])
    assert got.neighbor_mask.shape[1] <= 4 and got.neighbor_mask.any()
    point, no_b = build_model(cfg(),"point"), build_model(cfg(),"no_b")
    point.load_state_dict(no_b.state_dict())
    x=torch.randn(2,4)
    torch.testing.assert_close(point(x)["y_hat"], no_b(x)["y_hat"])

def test_optimizer_groups_schedule_and_synthetic_training(tmp_path):
    c=cfg(); state=prepare_training_state(c,"spatial",42,4)
    assert len(state.optimizer.param_groups)==2
    assert state.optimizer.param_groups[1]["lr"] == state.optimizer.param_groups[0]["lr"]*3
    scheduler=build_lr_scheduler(state.optimizer,c,n_train=4); assert scheduler._phase2_warmup_steps==2
    result=train_arm(c,"spatial",42,tmp_path/"run",checkpoint_dir=tmp_path/"weights",point_table=table(),device="cpu",geometry_table=geometry())
    assert result["status"]=="completed" and (tmp_path/"run"/"raw"/"history.csv").is_file()
    checkpoint=torch.load(tmp_path/"weights"/"formal_best.pt",map_location="cpu",weights_only=False)
    assert checkpoint["kind"]=="formal" and checkpoint["selection"]["kind"]=="formal" and checkpoint["input_dim"]==4 and checkpoint["hidden_dim"]==3
    paired=cfg(); paired["training"]["lr_schedule"]="constant"; paired["training"]["max_epochs"]=6
    paired["selection"]["formal_start_epoch"]=6; paired["selection"]["early_stop_count_start_epoch"]=16
    paired_result=train_arm(paired,"point",42,tmp_path/"paired_run",checkpoint_dir=tmp_path/"paired_weights",point_table=table(),device="cpu")
    assert (tmp_path/"paired_weights"/"warmup.pt").is_file()
    assert paired_result["warmup_checkpoint"]==str(tmp_path/"paired_weights"/"warmup.pt")

def test_search_quota_replication_and_spatial_split():
    c=cfg(); point=build_point_search_plan(c)
    assert len(point.candidates)==24 and [x.bucket for x in point.candidates].count("tpe")==16
    for i,candidate in enumerate(point.candidates): record_trial_result(candidate,seed=42,status="completed",patient_macro_pathway_pcc=float(i),patient_macro_z_mse_selection=float(24-i))
    assert len(replication_queue(point))==6
    spatial=build_spatial_search_plan(c,point.candidates[-3:])
    assert len(spatial.candidates)==24 and sum(x.spatial_scope=="restricted" for x in spatial.candidates)==18 and sum(x.spatial_scope=="joint" for x in spatial.candidates)==6
    restored=search_plan_from_dict(point.as_dict())
    extension=build_extension_plan(c,"spatial",spatial,point.candidates[-3:])
    assert restored.sampler_method==point.sampler_method and len(extension.candidates)==12
    assert sum(x.spatial_scope=="restricted" for x in extension.candidates)==9 and sum(x.spatial_scope=="joint" for x in extension.candidates)==3
    with pytest.raises(ValueError): build_extension_plan(c,"spatial",spatial,point.candidates[-3:])
