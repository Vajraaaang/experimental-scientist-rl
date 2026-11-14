"""Retained offline fault traces. No network, keys, or actual provider responses."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]

from experimentlab.envs import PilotConfig,ScientistEnv
from integrations.jev_client import JevClient,JevError
from integrations.jev_research import FixtureTransport,Journal,run_episode,verify_episode,canonical
from research_stage import UnfilteredBank,quadrature_family
from jev_rl_study import development_tasks,file_hash,write_new,receipt_stats


class FaultFixture(FixtureTransport):
    def __init__(self,behavior): self.behavior=behavior

    def __call__(self,request):
        if self.behavior=='timeout_after_assay' and request['state']['experiment_count']>=1:
            raise TimeoutError('fixture transport timeout')
        if self.behavior=='rate_limit':
            raise JevError('rate_limit',transport={'endpoint':'/v1/systemone','http_status':429,
                                                  'provider_request_id':'fixture-only-rate-limit','retry_after':1.})
        response=super().__call__(request)
        if self.behavior=='drift': response['model']='jev-fixture-drift'
        if self.behavior=='usage_overflow': response['usage']['input_tokens']=10**400
        if self.behavior=='immediate_stop' and 'next_action' in response['answers']:
            answer=response['answers']['next_action']
            answer['choice']='action_36'
            answer['probabilities']={key:float(key=='action_36') for key in answer['probabilities']}
        return response


def main():
    output=ROOT/'reports/jev-rl-m1/failure-examples'
    output.mkdir(exist_ok=False)
    config=PilotConfig(final_cultures=128)
    task=development_tasks(1,config)[0]
    bank=UnfilteredBank(quadrature_family(2),config)
    rows=[]
    for behavior in ('immediate_stop','timeout_after_assay','drift','usage_overflow','rate_limit'):
        client=JevClient(FaultFixture(behavior),expected_model='jev-fixture')
        path=output/(behavior+'.jsonl')
        with Journal(path) as journal:
            result=run_episode(ScientistEnv(task,config),'jev_only',bank,journal,client=client)
        replayed=verify_episode(path,task,config,bank)
        if canonical(result)!=canonical(replayed): raise ValueError('failure replay mismatch')
        rows.append({'behavior':behavior,'status':result['status'],'failure':result['failure'],
                     'assays':result['assays'],'variable_cost':result['variable_cost'],
                     'final_mse':result['final_mse'],'coverage':result['forecast_coverage'],
                     'api':receipt_stats(client.receipts),'journal_sha256':file_hash(path),
                     'verified':True})
    report={'scope':'offline injected faults; not provider behavior','hosted_calls':0,
            'driver_sha256':file_hash(Path(__file__).resolve()),'rows':rows}
    write_new(output/'summary.json',report)
    print(canonical(report))


if __name__=='__main__': main()
