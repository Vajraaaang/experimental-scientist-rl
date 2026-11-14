"""Bounded hardware development orchestration; no scientific configuration selection."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
PYTHON=ROOT/'.venv/bin/python'
CLI=ROOT/'scripts/profile_m3_ppo.py'
OUT=ROOT/'reports/jev-rl-m2/hardware-sweep'
WALL_CAP=540.  # plus <=60s reserved for three initial profiles, total <=600s
TRANSITION_CAP=2_950_000  # plus initial profiles remains below3M


def write(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def main():
    OUT.mkdir(exist_ok=False)
    start=time.perf_counter();completed=[];transitions=0
    planned=[]
    # Single-process samples are sequential to avoid cross-backend contention.
    for batch in (128,512,1024):
        for backend in ('cpu','mps','split'):
            planned.append(dict(name=f'joint-{backend}-b{batch}',backend=backend,batch=batch,
                                hidden=64,kind='recurrent',threads=1,workload='joint',initialization='v6',seconds=15,cycles=5))
    for threads in (2,4):
        planned.append(dict(name=f'cpu-threads{threads}',backend='cpu',batch=512,hidden=64,
                            kind='recurrent',threads=threads,workload='joint',initialization='v6',seconds=15,cycles=5))
    for kind,hidden in (('history',64),('history',128),('history',256),('recurrent',128),('recurrent',256)):
        for backend in ('cpu','mps','split'):
            planned.append(dict(name=f'selector-{kind}-h{hidden}-{backend}',backend=backend,batch=512,
                                hidden=hidden,kind=kind,threads=1,workload='selector',initialization='scratch',seconds=15,cycles=5))
    write(OUT/'plan.json',{'scope':'hardware-only; predeclared finite grid','active_wall_cap_seconds':WALL_CAP,
                         'transition_cap':TRANSITION_CAP,'initial_profile_reservation_seconds':60,
                         'driver_sha256':hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
                         'grid':planned,'sustained':'up to60s each cpu,mps,split at fastest correct measured batch',
                         'concurrency':'2 CPU seeds for30s and4 CPU seeds for45s; one thread each',
                         'thermal_claim':'OS status + throughput drift, not temperature attribution'})
    environment={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}

    def group(configs,seconds,cap_each):
        nonlocal transitions
        remaining=WALL_CAP-(time.perf_counter()-start)
        if remaining<10 or TRANSITION_CAP-transitions < 6000*len(configs): return False
        seconds=min(seconds,max(1.,remaining-8))
        cap_each=min(cap_each,(TRANSITION_CAP-transitions)//len(configs))
        jobs=[];tick=time.perf_counter()
        for index,c in enumerate(configs):
            destination=OUT/c['name'];log=(OUT/(c['name']+'.log')).open('x')
            cmd=[str(PYTHON),str(CLI),'--output',str(destination),'--seconds',str(seconds),
                 '--transition-cap',str(cap_each),'--checkpoint-every',str(c.get('checkpoint_every',1)),
                 '--seed',str(c.get('seed',20))]
            for key in ('backend','batch','hidden','kind','threads','workload','initialization','cycles'):
                cmd += ['--'+key,str(c[key])]
            process=subprocess.Popen(cmd,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
            jobs.append((process,log,destination,c))
        while any(p.poll() is None for p,_,_,_ in jobs):
            if time.perf_counter()-start>=WALL_CAP or time.perf_counter()-tick>seconds+8:
                for p,_,_,_ in jobs:
                    if p.poll() is None:p.terminate()
                break
            time.sleep(.2)
        for p,log,destination,c in jobs:
            try:code=p.wait(timeout=3)
            except subprocess.TimeoutExpired:p.kill();code=p.wait()
            log.close();path=destination/'result.json'
            if path.exists():
                result=json.loads(path.read_text());transitions += result['totals']['all_transitions']
                entry={'name':c['name'],'exit_code':code,'result':str(path),'config':result['config'],
                       'totals':result['totals'],'correctness':result['correctness'],'memory':result['memory'],
                       'sustained':result['sustained'],'stage_totals_seconds':result['stage_totals_seconds'],
                       'thermal':result['thermal'],'group_wall_seconds':time.perf_counter()-tick,'group_size':len(configs)}
            else:entry={'name':c['name'],'exit_code':code,'status':'incomplete','group_size':len(configs),
                        'possible_unreported_transitions_upper_bound':cap_each}
            completed.append(entry)
        write(OUT/'progress.json',{'completed':completed,'all_transitions':transitions,'elapsed':time.perf_counter()-start})
        print(json.dumps({'completed':[c['name'] for c in configs],'elapsed':time.perf_counter()-start,
                          'transitions':transitions}),flush=True)
        return True

    for config in planned:
        if not group([config],config['seconds'],30_000):break
    winners={}
    for backend in ('cpu','mps','split'):
        candidates=[r for r in completed if r.get('exit_code')==0 and r.get('totals',{}).get('cycles',0)>=3
                    and r['config']['backend']==backend and r['config']['workload']=='joint']
        if candidates:
            best=max(candidates,key=lambda r:r['totals']['training_transitions_per_cycle_second'])
            winners[backend]=best['config']
            c={**best['config'],'name':f'sustained-{backend}','seconds':60,'cycles':100_000,'checkpoint_every':50}
            group([c],60,550_000)
    if 'cpu' in winners:
        best=winners['cpu']
        for n,seconds in ((2,30),(4,45)):
            configs=[{**best,'name':f'concurrency{n}-seed{seed}','seed':seed,'threads':1,
                      'seconds':seconds,'cycles':100_000,'checkpoint_every':50} for seed in range(20,20+n)]
            group(configs,seconds,350_000)
    write(OUT/'summary.json',{'scope':'bounded hardware profile; no scientific selection','elapsed_seconds':time.perf_counter()-start,
        'global_active_wall_cap_seconds':WALL_CAP,'all_reported_transitions':transitions,'transition_cap':TRANSITION_CAP,
        'hardware_winners':winners,'runs':completed,'hosted_calls':0,'final_test_opened':False,
        'limitations':['initial profile times are separate','OS thermal status does not measure temperature',
                       'process RSS peaks are not a synchronized whole-machine unified-memory peak',
                       'v6 initialization uses base weights, not deployment public-price calibration']})


if __name__=='__main__':main()
