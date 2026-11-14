"""Matched complete CPU deployment timing on development tasks, after freeze."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from research_stage import UnfilteredBank,quadrature_family,unfiltered_tasks
from experimentlab.envs import PilotConfig,ScientistEnv
from experimentlab.tasks import continuous_family
from experimentlab.evaluation import run_strategy
from experimentlab.learning import ActorCritic,collect_episode
from experimentlab.cli import write_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1);study=ROOT/'reports/unfiltered-v3'
    bank=UnfilteredBank(quadrature_family(2),PilotConfig())
    tasks=unfiltered_tasks(continuous_family(8401,1024),bank,32,5432)
    schedule=tuple(json.loads((study/'exhaustive-fixed.json').read_text())['tuning']['schedule'])
    checkpoint=study/'converged-imitation/recurrent-imitation-seed0.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=ActorCritic(**state['architecture']);model.load_state_dict(state['model']);model.eval()
    def run(method,task):
        if method=='learned_imitation':collect_episode(model,ScientistEnv(task,bank.config),'cpu',deterministic=True)
        else:run_strategy(task,bank,'fixed' if method=='fixed' else 'voi',schedule if method=='fixed' else (),planner_draws=12,seed=10)
    records={}
    for method in ('fixed','voi','learned_imitation'):
        run(method,tasks[0])
        times=[]
        for _ in range(3):
            tick=time.perf_counter()
            for task in tasks:run(method,task)
            times.append((time.perf_counter()-tick)/len(tasks))
        records[method]={'seconds_per_complete_episode':times,'median_seconds':float(np.median(times))}
    result={'scope':'development_latency_only_no_test_reexecution','episodes_per_repeat':32,'repeats':3,
            'includes':['environment construction/reset','common baseline','all online inference','assays','final scoring'],
            'excludes':['prior cache construction','checkpoint loading','offline teacher and training'],
            'cache_seconds':bank.build_seconds,'policy_seed':0,'seed_selection':'fixed lowest seed, not best-performing seed',
            'measurements':records,'planner_to_learned_ratio':records['voi']['median_seconds']/records['learned_imitation']['median_seconds']}
    write_json(ROOT/'reports/online-compute.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
