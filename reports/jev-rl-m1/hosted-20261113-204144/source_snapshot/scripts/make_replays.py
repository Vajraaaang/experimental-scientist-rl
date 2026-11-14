"""Post-episode success/failure illustrations on development mechanisms."""
from pathlib import Path
import numpy as np
import torch
from research_stage import UnfilteredBank,quadrature_family,unfiltered_tasks
from experimentlab.cli import write_json
from experimentlab.envs import PilotConfig,ScientistEnv
from experimentlab.tasks import continuous_family
from experimentlab.evaluation import reference_result
from experimentlab.learning import ActorCritic,collect_episode

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1);config=PilotConfig()
    bank=UnfilteredBank(quadrature_family(2),config)
    tasks=unfiltered_tasks(continuous_family(8401,1024),bank,256,7532)
    checkpoint=ROOT/'reports/unfiltered-v3/strong-finetune/recurrent-imitation_ppo-strong-seed0.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=ActorCritic(**state['architecture']);model.load_state_dict(state['model']);model.eval()
    output=ROOT/'reports/replays';found={}
    for task in tasks:
        env=ScientistEnv(task,config);record=collect_episode(model,env,'cpu',deterministic=True)
        result=dict(env.private_result);result['return']=float(sum(record['rewards']))
        reference=reference_result(task,bank)
        if result['final_mse']<=.04**2:label='success'
        elif reference['final_mse']<=.04**2:label='learned-policy-failure'
        else:label='unreachable-target'
        if label in found:continue
        public=env.public_record();public['measurement_model']={'readout_sd':.02,'cohort_size':8,'culture_offset_sd':0.}
        payload={'scope':'development_illustration_selected_after_scoring_not_test_evaluation','public':public,
                 'post_episode_private_result':result,'post_episode_private_reference':reference}
        write_json(output/f'{label}.json',payload)
        lines=[f'# {label.replace("-"," ").title()} replay','',
               'Selected after scoring for illustration; this does not estimate a success rate. Private results appear only after the commitment.','',
               f"Public target X6/X7: {list(task.target)}. Variable budget: 8. Common baseline setup cost: 3.",'',
               '| Step | Suppression | Duration | Purchased panel | Readout mean +/- SD | Cost | Remaining |',
               '|---:|---|---:|---|---|---:|---:|']
        remaining=8.
        for step,packet in enumerate(public['trace'][1:],1):
            a=env.actions[packet['action_id']];remaining-=a.cost_ticks/20
            u=', '.join(f'X{i} at {v:g}' for i,v in enumerate(a.intervention) if v) or 'none'
            panel=list(np.flatnonzero(packet['measured_mask']))
            values=', '.join(f"X{i}: {packet['readouts'][:,i].mean():.4f} +/- {packet['readouts'][:,i].std():.4f}" for i in panel)
            lines.append(f'|{step}|{u}|{a.duration:g}|{panel}|{values}|{a.cost_ticks/20:.2f}|{remaining:.2f}|')
        final=env.actions[result['action_id']]
        u=', '.join(f'X{i} at {v:g}' for i,v in enumerate(final.intervention) if v) or 'none'
        lines.extend(['',f'Final commitment: {u} for time 6.','',
                      '## Post-episode private evaluator view','',
                      f"Final RMSE: {np.sqrt(result['final_mse']):.5f}; return {result['return']:.5f}; private nominal-reference RMSE {np.sqrt(reference['final_mse']):.5f}.",''])
        (output/f'{label}.md').write_text('\n'.join(lines));found[label]=True
        if len(found)==3:break
    write_json(output/'index.json',{'examples':list(found),'checkpoint':str(checkpoint),'locked_test_reexecuted':False})
    print('Rendered development replay examples:',list(found))


if __name__=='__main__':main()
