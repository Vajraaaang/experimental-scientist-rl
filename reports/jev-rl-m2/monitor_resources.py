"""Nonprivileged sampled memory/thermal observations during remaining race work."""
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/jev-rl-m2'
RACE=OUT/'race-01'


def output(arguments):
    result=subprocess.run(arguments,capture_output=True,text=True,timeout=3)
    return result.stdout.strip()


def sample():
    memory=int(output(['/usr/sbin/sysctl','-n','hw.memsize']))
    raw=output(['/usr/bin/vm_stat'])
    page_size=int(re.search(r'page size of (\d+) bytes',raw).group(1))
    pages={m.group(1):int(m.group(2)) for m in re.finditer(r'^([^:\n]+):\s+(\d+)\.',raw,re.M)}
    thermal=int(output(['/usr/bin/osascript','-l','JavaScript','-e',
                        'ObjC.import("Foundation"); Number($.NSProcessInfo.processInfo.thermalState)']))
    pattern=re.escape(str(ROOT/'scripts/train_ppo_m2.py'))+r'.*race-01'
    pids=output(['/usr/bin/pgrep','-f',pattern]).splitlines()
    processes=[]
    for pid in pids:
        text=output(['/bin/ps','-p',pid,'-o','pid=,rss=,pcpu=']).split()
        if len(text)==3:processes.append({'pid':int(text[0]),'rss_bytes':int(text[1])*1024,'cpu_percent':float(text[2])})
    return {'utc':datetime.now(timezone.utc).isoformat(),'thermal_state_code':thermal,
            'physical_unified_pool_bytes':memory,'vm_page_size':page_size,'vm_pages':pages,
            'nonfree_pool_bytes_including_other_apps_and_cache':memory-pages.get('Pages free',0)*page_size,
            'owned_training_processes':processes,'sum_owned_training_rss_bytes':sum(p['rss_bytes'] for p in processes)}


def main():
    started=time.perf_counter();rows=[]
    path=OUT/'resource-samples.jsonl'
    with path.open('x') as stream:
        while True:
            row=sample();rows.append(row);stream.write(json.dumps(row,sort_keys=True)+'\n');stream.flush()
            if (RACE/'selection.json').exists() or time.perf_counter()-started>=1200:break
            time.sleep(15)
    summary={'scope':'sampled remaining-race window; not full-race or exact allocator peak',
             'samples':len(rows),'started_after_race_seconds_approx':max(0,time.time()-(RACE/'plan.json').stat().st_mtime-(time.perf_counter()-started)),
             'seconds':time.perf_counter()-started,'interval_seconds':15,
             'sampled_peak_owned_training_rss_bytes':max(r['sum_owned_training_rss_bytes'] for r in rows),
             'sampled_peak_nonfree_unified_pool_bytes':max(r['nonfree_pool_bytes_including_other_apps_and_cache'] for r in rows),
             'thermal_state_codes':sorted(set(r['thermal_state_code'] for r in rows)),
             'thermal_code_meanings':{'0':'nominal','1':'fair','2':'serious','3':'critical'},
             'temperature_celsius_measured':False,
             'limitations':['nonfree machine memory includes caches and unrelated apps','RSS sums may share pages',
                            '15-second sampling can miss shorter peaks','monitor starts after initial profiling/training; no retrospective peak claim']}
    (OUT/'resource-summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
