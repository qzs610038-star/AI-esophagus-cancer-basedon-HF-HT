"""Deterministic search planning and result bookkeeping for UNI2-h only.

Plans deliberately separate the phase-start sampler decision from subsequent
trial execution: a phase either remains Optuna TPE throughout, or remains the
pre-generated stratified fallback throughout.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import ast
import importlib.metadata
import math
from typing import Any, Iterable
import numpy as np

from errors import ConfigError

@dataclass
class SearchCandidate:
    candidate_id: str
    stage: str
    bucket: str                 # anchor/random/tpe
    spatial_scope: str          # point/restricted/joint
    parameters: dict[str, Any]
    head_source_id: str | None = None
    status: str = "pending"
    results: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict: return asdict(self)

@dataclass
class SearchStagePlan:
    stage: str
    sampler_method: str         # fixed at phase start
    sampler_seed: int
    candidates: list[SearchCandidate]
    replication_seeds: list[int]
    top_k: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["candidate_count"] = len(self.candidates)
        return payload

def search_plan_from_dict(payload: dict) -> SearchStagePlan:
    """Restore a saved plan without reseeding or changing its sampler decision."""
    required={"stage","sampler_method","sampler_seed","candidates","replication_seeds","top_k"}
    missing=required-set(payload)
    if missing: raise ConfigError(f"搜索计划缺少字段 {sorted(missing)}")
    candidates=[]
    for row in payload["candidates"]:
        if not isinstance(row,dict): raise ConfigError("搜索候选必须为 object")
        candidates.append(SearchCandidate(**{key:row[key] for key in SearchCandidate.__dataclass_fields__ if key in row}))
    ids=[c.candidate_id for c in candidates]
    if len(ids)!=len(set(ids)): raise ConfigError("搜索计划 candidate_id 重复")
    return SearchStagePlan(str(payload["stage"]),str(payload["sampler_method"]),int(payload["sampler_seed"]),candidates,
                           [int(v) for v in payload["replication_seeds"]],int(payload["top_k"]),dict(payload.get("metadata") or {}))

def resolve_sampling_method(search: dict) -> str:
    """Choose exactly once before a stage starts; do not switch after failures."""
    expected = str(search.get("optuna_version", "5.0.0"))
    if str(search.get("sampler", "optuna_tpe")) != "optuna_tpe":
        return "stratified_random"
    try:
        actual = importlib.metadata.version("optuna")
        if actual != expected:
            return "stratified_random"
        import optuna
        if str(optuna.__version__) != expected or not hasattr(optuna.samplers, "TPESampler"):
            return "stratified_random"
        return "optuna_tpe"
    except (ImportError, OSError, AttributeError, importlib.metadata.PackageNotFoundError):
        return "stratified_random"

def _choice(rng: np.random.Generator, values: list[Any]) -> Any: return values[int(rng.integers(len(values)))]
def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float: return float(math.exp(rng.uniform(math.log(low), math.log(high))))

def _head_parameters(search: dict, rng: np.random.Generator, *, stratified_index: int | None = None) -> dict[str, Any]:
    optimizer = _choice(rng, list(search["optimizer_choices"]))
    if stratified_index is not None:
        # Each discrete axis is cycled so the fallback cannot accidentally omit it.
        optimizer = list(search["optimizer_choices"])[stratified_index % len(search["optimizer_choices"])]
        batch = list(search["batch_size_choices"])[stratified_index % len(search["batch_size_choices"])]
        hidden = list(search["hidden_dim_choices"])[stratified_index % len(search["hidden_dim_choices"])]
        dropout = list(search["dropout_choices"])[stratified_index % len(search["dropout_choices"])]
        schedule = list(search["lr_schedule_choices"])[stratified_index % len(search["lr_schedule_choices"])]
    else:
        batch, hidden = _choice(rng, list(search["batch_size_choices"])), _choice(rng, list(search["hidden_dim_choices"]))
        dropout, schedule = _choice(rng, list(search["dropout_choices"])), _choice(rng, list(search["lr_schedule_choices"]))
    params = {"training.optimizer": optimizer, "training.learning_rate": _log_uniform(rng, *search["learning_rate_log_range"]),
              "training.batch_size": batch, "model.hidden_dim": hidden, "model.dropout": dropout, "training.lr_schedule": schedule,
              "training.b_lr_multiplier": _choice(rng, list(search["b_lr_multiplier_choices"]))}
    params["training.weight_decay"] = 0.0 if optimizer == "Adam" else _log_uniform(rng, *search["adamw_weight_decay_log_range"])
    return params

def _graph_parameters(search: dict, rng: np.random.Generator, *, stratified_index: int | None = None) -> dict[str, Any]:
    pairs = list(search["graph_radius_neighbor_choices"])
    pair = pairs[(stratified_index % len(pairs)) if stratified_index is not None else int(rng.integers(len(pairs)))]
    return {"graph.radius_in_native_steps": float(pair[0]), "graph.max_neighbors": int(pair[1]),
            "graph.distance_sigma": _choice(rng, list(search["graph_distance_sigma_choices"])),
            "graph.image_temperature": _log_uniform(rng, *search["graph_image_temperature_log_range"]),
            "graph.self_raw_weight": _choice(rng, list(search["graph_self_raw_weight_choices"]))}

def _anchors() -> list[dict[str, Any]]:
    return [
        {"training.optimizer": "AdamW", "training.learning_rate": 3e-4, "training.weight_decay": 1e-4, "training.batch_size": 256, "model.hidden_dim": 256, "model.dropout": .3, "training.lr_schedule": "constant", "training.b_lr_multiplier": 1.0},
        {"training.optimizer": "Adam", "training.learning_rate": 1e-4, "training.weight_decay": 0.0, "training.batch_size": 32, "model.hidden_dim": 256, "model.dropout": .3, "training.lr_schedule": "constant", "training.b_lr_multiplier": 1.0},
    ]

def build_point_search_plan(config: dict) -> SearchStagePlan:
    search = config["search"]; rng = np.random.default_rng(int(search["point_sampler_seed"])); method = resolve_sampling_method(search)
    items=[]
    for i, params in enumerate(_anchors()): items.append(SearchCandidate(f"point-{i+1:02d}", "point", "anchor", "point", params))
    for i in range(int(search["random_trials"])): items.append(SearchCandidate(f"point-{len(items)+1:02d}", "point", "random", "point", _head_parameters(search, rng, stratified_index=i)))
    for i in range(int(search["tpe_trials"])):
        # The fallback values are generated at phase start and remain usable if Optuna is unavailable.
        items.append(SearchCandidate(f"point-{len(items)+1:02d}", "point", "tpe", "point", _head_parameters(search, rng, stratified_index=i if method == "stratified_random" else None)))
    _assert_initial(search, items)
    return SearchStagePlan("point", method, int(search["point_sampler_seed"]), items, list(search["replication_seeds"]), int(search["top_k"]), {"model_name": "uni2h"})

def build_spatial_search_plan(config: dict, point_top3: Iterable[SearchCandidate | dict]) -> SearchStagePlan:
    search = config["search"]; heads = list(point_top3)
    if len(heads) < 3: raise ConfigError("空间搜索需要已经冻结的单点前三配置")
    def item_params(v): return v.parameters if isinstance(v, SearchCandidate) else dict(v["parameters"])
    def item_id(v): return v.candidate_id if isinstance(v, SearchCandidate) else str(v["candidate_id"])
    rng=np.random.default_rng(int(search["spatial_sampler_seed"])); method=resolve_sampling_method(search); items=[]
    # Two anchors use the best two point heads and original graph parameters.
    base_graph={"graph.radius_in_native_steps":1.5,"graph.max_neighbors":8,"graph.distance_sigma":1.0,"graph.image_temperature":.2,"graph.self_raw_weight":1.0}
    for i, head in enumerate(heads[:2]): items.append(SearchCandidate(f"spatial-{len(items)+1:02d}","spatial","anchor","restricted",{**item_params(head),**base_graph},item_id(head)))
    # 18 restricted total (anchors included); 6 joint total. Random is 4/2, TPE is 12/4.
    for i in range(4):
        head=heads[i%3]; items.append(SearchCandidate(f"spatial-{len(items)+1:02d}","spatial","random","restricted",{**item_params(head),**_graph_parameters(search,rng,stratified_index=i)},item_id(head)))
    for i in range(2):
        items.append(SearchCandidate(f"spatial-{len(items)+1:02d}","spatial","random","joint",{**_head_parameters(search,rng,stratified_index=i),**_graph_parameters(search,rng,stratified_index=i)},None))
    for i in range(12):
        head=heads[i%3]; items.append(SearchCandidate(f"spatial-{len(items)+1:02d}","spatial","tpe","restricted",{**item_params(head),**_graph_parameters(search,rng,stratified_index=i if method=="stratified_random" else None)},item_id(head)))
    for i in range(4):
        items.append(SearchCandidate(f"spatial-{len(items)+1:02d}","spatial","tpe","joint",{**_head_parameters(search,rng,stratified_index=i if method=="stratified_random" else None),**_graph_parameters(search,rng,stratified_index=i if method=="stratified_random" else None)},None))
    _assert_initial(search, items)
    if sum(c.spatial_scope == "restricted" for c in items) != 18 or sum(c.spatial_scope == "joint" for c in items) != 6: raise AssertionError("空间搜索名额错误")
    return SearchStagePlan("spatial",method,int(search["spatial_sampler_seed"]),items,list(search["replication_seeds"]),int(search["top_k"]),{"model_name":"uni2h","point_top3":[item_id(h) for h in heads[:3]]})

def build_extension_plan(config: dict, stage: str, base_plan: SearchStagePlan | dict, point_top3: Iterable[SearchCandidate | dict] | None = None) -> SearchStagePlan:
    """Create the explicitly requested one-off twelve-candidate extension.

    The new plan inherits the sampler choice already written for the original
    phase.  It does not re-probe Optuna; it records extension_count on the
    supplied plan so a caller that persists it cannot issue a second one.
    """
    if isinstance(base_plan, dict): base_plan=search_plan_from_dict(base_plan)
    if stage not in {"point","spatial"} or base_plan.stage != stage: raise ConfigError("扩展阶段必须与原搜索阶段一致")
    if int(base_plan.metadata.get("extension_count",0)) >= int(config["search"]["extension_max_per_stage"]):
        raise ConfigError("每个搜索阶段最多显式扩展一次")
    search=config["search"]; n=int(search["extension_trials"])
    if n != 12: raise ConfigError("本协议的显式扩展固定为 12 个初搜候选")
    rng=np.random.default_rng(base_plan.sampler_seed + 1000003); items=[]
    if stage=="point":
        for i in range(n):
            items.append(SearchCandidate(f"point-ext-{i+1:02d}","point","extension","point",_head_parameters(search,rng,stratified_index=i)))
    else:
        heads=list(point_top3 or [])
        if len(heads)<3: raise ConfigError("空间扩展需要原单点前三配置")
        def params(v): return v.parameters if isinstance(v,SearchCandidate) else dict(v["parameters"])
        def ident(v): return v.candidate_id if isinstance(v,SearchCandidate) else str(v["candidate_id"])
        # The extension preserves the original 18:6 intent at the 12-item scale as 9 restricted + 3 joint.
        for i in range(9):
            head=heads[i%3]; items.append(SearchCandidate(f"spatial-ext-{i+1:02d}","spatial","extension","restricted",{**params(head),**_graph_parameters(search,rng,stratified_index=i)},ident(head)))
        for i in range(3):
            items.append(SearchCandidate(f"spatial-ext-{i+10:02d}","spatial","extension","joint",{**_head_parameters(search,rng,stratified_index=i),**_graph_parameters(search,rng,stratified_index=i)}))
    extension_count=int(base_plan.metadata.get("extension_count",0))+1
    base_plan.metadata["extension_count"]=extension_count
    return SearchStagePlan(stage,base_plan.sampler_method,base_plan.sampler_seed,items,list(base_plan.replication_seeds),base_plan.top_k,
                           {"model_name":"uni2h","extension_of":base_plan.stage,"extension_count":extension_count,"explicit_only":True,"initial_candidate_count":n,"extension_generation":"pre_generated_stratified_random"})

def next_tpe_candidate(plan: SearchStagePlan, config: dict) -> SearchCandidate | None:
    """Fill one pending TPE slot from completed seed-42 observations.

    Callers invoke this immediately before dispatching each TPE trial.  It
    cannot change a phase whose `sampler_method` was fixed to fallback.
    """
    candidate = next((c for c in plan.candidates if c.bucket == "tpe" and c.status == "pending"), None)
    if candidate is None or plan.sampler_method != "optuna_tpe": return candidate
    try:
        import optuna
        from optuna.distributions import CategoricalDistribution, FloatDistribution
    except ImportError as exc:
        raise RuntimeError("阶段已锁定 Optuna TPE 但运行环境随后丢失 Optuna；不得中途切换") from exc
    search=config["search"]; sampler=optuna.samplers.TPESampler(seed=plan.sampler_seed, multivariate=False)
    study=optuna.create_study(direction="maximize",sampler=sampler)
    distributions={"optimizer":CategoricalDistribution(list(search["optimizer_choices"])), "learning_rate":FloatDistribution(*search["learning_rate_log_range"],log=True),
                   "batch_size":CategoricalDistribution(list(search["batch_size_choices"])), "hidden_dim":CategoricalDistribution(list(search["hidden_dim_choices"])),
                   "dropout":CategoricalDistribution(list(search["dropout_choices"])), "lr_schedule":CategoricalDistribution(list(search["lr_schedule_choices"])),
                   "b_lr_multiplier":CategoricalDistribution(list(search["b_lr_multiplier_choices"])), "weight_decay":FloatDistribution(*search["adamw_weight_decay_log_range"],log=True),
                   "radius_neighbor":CategoricalDistribution([str(v) for v in search["graph_radius_neighbor_choices"]]), "distance_sigma":CategoricalDistribution(list(search["graph_distance_sigma_choices"])),
                   "image_temperature":FloatDistribution(*search["graph_image_temperature_log_range"],log=True), "self_raw_weight":CategoricalDistribution(list(search["graph_self_raw_weight_choices"]))}
    for previous in plan.candidates:
        results={r["seed"]:r for r in previous.results if r["status"]=="completed"}
        if 42 not in results: continue
        p=previous.parameters
        # Only fully specified trials can become Optuna observations.
        required=("training.optimizer","training.learning_rate","training.batch_size","model.hidden_dim","model.dropout","training.lr_schedule","training.b_lr_multiplier")
        if not all(k in p for k in required): continue
        pair=[p.get("graph.radius_in_native_steps",1.5),p.get("graph.max_neighbors",8)]
        values={"optimizer":p["training.optimizer"],"learning_rate":p["training.learning_rate"],"batch_size":p["training.batch_size"],"hidden_dim":p["model.hidden_dim"],"dropout":p["model.dropout"],"lr_schedule":p["training.lr_schedule"],"b_lr_multiplier":p["training.b_lr_multiplier"],"weight_decay":p.get("training.weight_decay",1e-4),"radius_neighbor":str(pair),"distance_sigma":p.get("graph.distance_sigma",1.),"image_temperature":p.get("graph.image_temperature",.2),"self_raw_weight":p.get("graph.self_raw_weight",1.)}
        # Adam has no weight-decay parameter.  Keeping its zero in a positive
        # log distribution would silently discard every completed Adam trial.
        if values["optimizer"] == "Adam":
            values.pop("weight_decay")
        observed_distributions = {key: distributions[key] for key in values}
        try: study.add_trial(optuna.trial.create_trial(params=values, distributions=observed_distributions, value=float(results[42]["patient_macro_pathway_pcc"])))
        except ValueError: pass
    trial=study.ask()
    optimizer=trial.suggest_categorical("optimizer",list(search["optimizer_choices"])); params={"training.optimizer":optimizer,"training.learning_rate":trial.suggest_float("learning_rate",*search["learning_rate_log_range"],log=True),"training.batch_size":trial.suggest_categorical("batch_size",list(search["batch_size_choices"])),"model.hidden_dim":trial.suggest_categorical("hidden_dim",list(search["hidden_dim_choices"])),"model.dropout":trial.suggest_categorical("dropout",list(search["dropout_choices"])),"training.lr_schedule":trial.suggest_categorical("lr_schedule",list(search["lr_schedule_choices"])),"training.b_lr_multiplier":trial.suggest_categorical("b_lr_multiplier",list(search["b_lr_multiplier_choices"]))}
    params["training.weight_decay"] = 0.0 if optimizer=="Adam" else trial.suggest_float("weight_decay",*search["adamw_weight_decay_log_range"],log=True)
    if plan.stage=="spatial":
        # Restricted spatial slots only search graph values.  Their H/C/B
        # training recipe remains the complete parameter set of one point top-3.
        if candidate.spatial_scope == "restricted":
            params = {key: value for key, value in candidate.parameters.items() if not key.startswith("graph.")}
        pair=ast.literal_eval(trial.suggest_categorical("radius_neighbor",[str(v) for v in search["graph_radius_neighbor_choices"]]))
        params.update({"graph.radius_in_native_steps":float(pair[0]),"graph.max_neighbors":int(pair[1]),"graph.distance_sigma":trial.suggest_categorical("distance_sigma",list(search["graph_distance_sigma_choices"])),"graph.image_temperature":trial.suggest_float("image_temperature",*search["graph_image_temperature_log_range"],log=True),"graph.self_raw_weight":trial.suggest_categorical("self_raw_weight",list(search["graph_self_raw_weight_choices"]))})
    candidate.parameters=params
    return candidate

def _assert_initial(search: dict, items: list[SearchCandidate]) -> None:
    if len(items) != int(search["initial_trials"]) or sum(x.bucket=="anchor" for x in items)!=2 or sum(x.bucket=="random" for x in items)!=6 or sum(x.bucket=="tpe" for x in items)!=16: raise AssertionError("初搜必须严格为2锚点+6随机+16TPE")

def record_trial_result(candidate: SearchCandidate, *, seed: int, status: str, patient_macro_pathway_pcc: float | None = None, patient_macro_z_mse_selection: float | None = None, run_dir: str | None = None, error: str | None = None) -> SearchCandidate:
    if status not in {"completed", "failed"}: raise ValueError("status 必须为 completed/failed")
    if any(int(x["seed"]) == int(seed) for x in candidate.results): raise ValueError("同一候选和种子不能重复登记")
    candidate.results.append({"seed":int(seed),"status":status,"patient_macro_pathway_pcc":patient_macro_pathway_pcc,"patient_macro_z_mse_selection":patient_macro_z_mse_selection,"run_dir":run_dir,"error":error})
    candidate.status = "failed" if status == "failed" else "completed"
    return candidate

def rank_candidates(candidates: Iterable[SearchCandidate], *, seeds: Iterable[int] = (42,), tolerance: float = 1e-6) -> list[SearchCandidate]:
    required=set(map(int,seeds)); eligible=[]
    for c in candidates:
        got={int(r["seed"]):r for r in c.results if r["status"]=="completed"}
        if not required.issubset(got): continue
        score=float(np.mean([float(got[s]["patient_macro_pathway_pcc"]) for s in required])); mse=float(np.mean([float(got[s]["patient_macro_z_mse_selection"]) for s in required]))
        eligible.append((c,score,mse))
    eligible.sort(key=lambda x: (-round(x[1]/tolerance) if tolerance else -x[1], x[2], x[0].candidate_id))
    # Sorting using buckets above is only a coarse grouping; correct PCC tolerance comparison explicitly.
    from functools import cmp_to_key
    def cmp(a,b):
        d=a[1]-b[1]
        if abs(d)>tolerance: return -1 if d>0 else 1
        if a[2]!=b[2]: return -1 if a[2]<b[2] else 1
        return -1 if a[0].candidate_id<b[0].candidate_id else (a[0].candidate_id>b[0].candidate_id)
    eligible.sort(key=cmp_to_key(cmp)); return [x[0] for x in eligible]

def replication_queue(plan: SearchStagePlan) -> list[tuple[str,int]]:
    top=rank_candidates(plan.candidates,seeds=[plan.replication_seeds[0]])[:plan.top_k]
    return [(c.candidate_id, seed) for c in top for seed in plan.replication_seeds[1:]]

def should_offer_extension(plan: SearchStagePlan, *, best_scores_in_order: list[float], best_parameters: dict[str,Any], config: dict, prior_extensions: int = 0) -> bool:
    search=config["search"]
    if prior_extensions >= int(search["extension_max_per_stage"]): return False
    if bool(search.get("extension_edge_trigger", True)):
        ranges={"training.learning_rate":search["learning_rate_log_range"],"training.weight_decay":search["adamw_weight_decay_log_range"],"graph.image_temperature":search["graph_image_temperature_log_range"]}
        for key,bounds in ranges.items():
            if key in best_parameters and np.isclose(float(best_parameters[key]),float(bounds[0])) or key in best_parameters and np.isclose(float(best_parameters[key]),float(bounds[1])): return True
    return len(best_scores_in_order)>=6 and float(best_scores_in_order[-1])-float(best_scores_in_order[-6]) > float(search["extension_last_six_gain"])
