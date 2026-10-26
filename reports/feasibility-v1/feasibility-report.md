# Measured feasibility report

This is a bounded synthetic development pilot. Locked test results have not been opened.

The two-mechanism diagnostic and branching toy establish implementation controls, not adaptive RL value.

| Budget | Noise SD | Offset SD | Cost weight | Method | RMSE | Variable cost | Return | Online s/episode |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
|8|0.02|0.0|0.01|prior|0.08000|0.000|-0.64000|0.00077|
|8|0.02|0.0|0.01|fixed|0.02891|4.600|-0.16956|0.00170|
|8|0.02|0.0|0.01|identification|0.05777|5.800|-0.42080|0.00216|
|8|0.02|0.0|0.01|additive|0.06283|5.800|-0.47296|0.00220|
|8|0.02|0.0|0.01|random|0.05460|7.129|-0.40424|0.00214|
|8|0.02|0.0|0.01|eig|0.02891|6.479|-0.18835|0.05249|
|8|0.02|0.0|0.01|voi|0.02891|3.800|-0.16156|0.04197|
|8|0.02|0.0|0.01|reference|0.02891|0.000|-0.12356|0.00000|
|8|0.08|0.0|0.01|prior|0.08000|0.000|-0.64000|0.00079|
|8|0.08|0.0|0.01|fixed|0.04106|4.800|-0.25578|0.00176|
|8|0.08|0.0|0.01|identification|0.05943|5.800|-0.44026|0.00221|
|8|0.08|0.0|0.01|additive|0.06786|5.800|-0.54450|0.00229|
|8|0.08|0.0|0.01|random|0.06471|7.129|-0.52022|0.00220|
|8|0.08|0.0|0.01|eig|0.03584|7.698|-0.24448|0.04623|
|8|0.08|0.0|0.01|voi|0.03104|4.948|-0.18582|0.04075|
|8|0.08|0.0|0.01|reference|0.02891|0.000|-0.12356|0.00000|
|8|10.0|0.0|0.01|prior|0.08000|0.000|-0.64000|0.00078|
|8|10.0|0.0|0.01|fixed|0.08000|0.000|-0.64000|0.00078|
|8|10.0|0.0|0.01|identification|0.08000|5.800|-0.69800|0.00222|
|8|10.0|0.0|0.01|additive|0.08320|5.800|-0.77289|0.00225|
|8|10.0|0.0|0.01|random|0.07983|7.129|-0.70871|0.00222|
|8|10.0|0.0|0.01|eig|0.08000|0.000|-0.64000|0.01866|
|8|10.0|0.0|0.01|voi|0.08000|0.000|-0.64000|0.01764|
|8|10.0|0.0|0.01|reference|0.02891|0.000|-0.12356|0.00000|
|8|0.02|0.03|0.01|prior|0.07955|0.000|-0.65113|0.00082|
|8|0.02|0.03|0.01|fixed|0.03016|4.600|-0.17697|0.00182|
|8|0.02|0.03|0.01|identification|0.06295|5.800|-0.48932|0.00228|
|8|0.02|0.03|0.01|additive|0.06373|5.800|-0.48520|0.00230|
|8|0.02|0.03|0.01|random|0.05459|7.129|-0.40552|0.00229|
|8|0.02|0.03|0.01|eig|0.03016|6.790|-0.19884|0.05419|
|8|0.02|0.03|0.01|voi|0.03016|3.677|-0.16770|0.04355|
|8|0.02|0.03|0.01|reference|0.03015|0.000|-0.13093|0.00000|
|8|0.02|0.0|0.005|prior|0.08000|0.000|-0.64000|0.00079|
|8|0.02|0.0|0.005|fixed|0.02891|4.600|-0.14656|0.00174|
|8|0.02|0.0|0.005|identification|0.05777|5.800|-0.39180|0.00220|
|8|0.02|0.0|0.005|additive|0.06283|5.800|-0.44396|0.00225|
|8|0.02|0.0|0.005|random|0.05460|7.129|-0.36859|0.00221|
|8|0.02|0.0|0.005|eig|0.02891|6.503|-0.15607|0.05531|
|8|0.02|0.0|0.005|voi|0.02891|3.818|-0.14265|0.04456|
|8|0.02|0.0|0.005|reference|0.02891|0.000|-0.12356|0.00000|
|8|0.02|0.0|0.02|prior|0.08000|0.000|-0.64000|0.00079|
|8|0.02|0.0|0.02|fixed|0.02891|4.600|-0.21556|0.00174|
|8|0.02|0.0|0.02|identification|0.05777|5.800|-0.47880|0.00221|
|8|0.02|0.0|0.02|additive|0.06283|5.800|-0.53096|0.00226|
|8|0.02|0.0|0.02|random|0.05460|7.129|-0.47553|0.00223|
|8|0.02|0.0|0.02|eig|0.02891|6.458|-0.25271|0.05511|
|8|0.02|0.0|0.02|voi|0.02891|3.798|-0.19953|0.04422|
|8|0.02|0.0|0.02|reference|0.02891|0.000|-0.12356|0.00000|
|16|0.02|0.0|0.01|prior|0.08000|0.000|-0.64000|0.00079|
|16|0.02|0.0|0.01|fixed|0.02891|4.600|-0.16956|0.00174|
|16|0.02|0.0|0.01|identification|0.02891|11.600|-0.23956|0.00363|
|16|0.02|0.0|0.01|additive|0.02891|11.600|-0.23956|0.00372|
|16|0.02|0.0|0.01|random|0.04901|9.211|-0.36840|0.00273|
|16|0.02|0.0|0.01|eig|0.02891|7.640|-0.19996|0.07202|
|16|0.02|0.0|0.01|voi|0.02891|3.800|-0.16156|0.04391|
|16|0.02|0.0|0.01|reference|0.02891|0.000|-0.12356|0.00000|
|16|0.08|0.0|0.01|prior|0.08000|0.000|-0.64000|0.00078|
|16|0.08|0.0|0.01|fixed|0.04106|4.800|-0.25578|0.00172|
|16|0.08|0.0|0.01|identification|0.02999|11.600|-0.24594|0.00361|
|16|0.08|0.0|0.01|additive|0.03431|11.600|-0.27105|0.00367|
|16|0.08|0.0|0.01|random|0.06036|9.211|-0.48895|0.00265|
|16|0.08|0.0|0.01|eig|0.02999|10.591|-0.23586|0.07298|
|16|0.08|0.0|0.01|voi|0.02891|5.069|-0.17425|0.04746|
|16|0.08|0.0|0.01|reference|0.02891|0.000|-0.12356|0.00000|
|16|10.0|0.0|0.01|prior|0.08000|0.000|-0.64000|0.00079|
|16|10.0|0.0|0.01|fixed|0.08000|0.000|-0.64000|0.00078|
|16|10.0|0.0|0.01|identification|0.08000|11.600|-0.75600|0.00365|
|16|10.0|0.0|0.01|additive|0.08457|11.600|-0.85568|0.00372|
|16|10.0|0.0|0.01|random|0.07983|9.211|-0.72953|0.00270|
|16|10.0|0.0|0.01|eig|0.08000|0.000|-0.64000|0.01900|
|16|10.0|0.0|0.01|voi|0.08000|0.000|-0.64000|0.01805|
|16|10.0|0.0|0.01|reference|0.02891|0.000|-0.12356|0.00000|
|16|0.02|0.03|0.01|prior|0.07955|0.000|-0.65113|0.00082|
|16|0.02|0.03|0.01|fixed|0.03016|4.600|-0.17697|0.00181|
|16|0.02|0.03|0.01|identification|0.03016|11.600|-0.24693|0.00373|
|16|0.02|0.03|0.01|additive|0.03016|11.600|-0.24698|0.00382|
|16|0.02|0.03|0.01|random|0.04871|9.211|-0.36702|0.00276|
|16|0.02|0.03|0.01|eig|0.03016|7.970|-0.21063|0.07483|
|16|0.02|0.03|0.01|voi|0.03016|3.677|-0.16770|0.04443|
|16|0.02|0.03|0.01|reference|0.03015|0.000|-0.13093|0.00000|
|16|0.02|0.0|0.005|prior|0.08000|0.000|-0.64000|0.00078|
|16|0.02|0.0|0.005|fixed|0.02891|4.600|-0.14656|0.00177|
|16|0.02|0.0|0.005|identification|0.02891|11.600|-0.18156|0.00365|
|16|0.02|0.0|0.005|additive|0.02891|11.600|-0.18156|0.00369|
|16|0.02|0.0|0.005|random|0.04901|9.211|-0.32234|0.00269|
|16|0.02|0.0|0.005|eig|0.02891|7.680|-0.16196|0.07410|
|16|0.02|0.0|0.005|voi|0.02891|3.818|-0.14265|0.04504|
|16|0.02|0.0|0.005|reference|0.02891|0.000|-0.12356|0.00000|
|16|0.02|0.0|0.02|prior|0.08000|0.000|-0.64000|0.00078|
|16|0.02|0.0|0.02|fixed|0.02891|4.600|-0.21556|0.00174|
|16|0.02|0.0|0.02|identification|0.02891|11.600|-0.35556|0.00363|
|16|0.02|0.0|0.02|additive|0.02891|11.600|-0.35556|0.00370|
|16|0.02|0.0|0.02|random|0.04901|9.211|-0.46050|0.00267|
|16|0.02|0.0|0.02|eig|0.02891|7.631|-0.27618|0.07405|
|16|0.02|0.0|0.02|voi|0.02891|3.798|-0.19953|0.04500|
|16|0.02|0.0|0.02|reference|0.02891|0.000|-0.12356|0.00000|

## Gate evidence

Finite-support development episodes share possible hidden mechanisms. Mechanism-group bootstrap intervals quantify this support only; they do not establish continuous or structural generalization.

- budget8-noise0.02-offset0.0-cost0.01: information gate B=True; finite-support C screen=False; raw gap recovery=0.000; adaptive-minus-fixed return {'mean': 0.007892485119047617, 'ci95': [0.005999808407738093, 0.009809990699404757], 'unit': 'mechanism', 'units': 32}; error reduction {'mean': 0.0, 'ci95': [0.0, 0.0], 'unit': 'mechanism', 'units': 32}.
- budget8-noise0.08-offset0.0-cost0.01: information gate B=True; finite-support C screen=True; raw gap recovery=0.130; adaptive-minus-fixed return {'mean': 0.06686994925634465, 'ci95': [0.02305236114930795, 0.12192996249212784], 'unit': 'mechanism', 'units': 32}; error reduction {'mean': 0.0006903293288729702, 'ci95': [0.00024678534844141533, 0.0012447640555334743], 'unit': 'mechanism', 'units': 32}.
- budget8-noise10.0-offset0.0-cost0.01: information gate B=False; finite-support C screen=False; raw gap recovery=0.000; adaptive-minus-fixed return {'mean': 0.0, 'ci95': [0.0, 0.0], 'unit': 'mechanism', 'units': 32}; error reduction {'mean': 0.0, 'ci95': [0.0, 0.0], 'unit': 'mechanism', 'units': 32}.
- budget8-noise0.02-offset0.03-cost0.01: information gate B=True; finite-support C screen=False; raw gap recovery=0.000; adaptive-minus-fixed return {'mean': 0.008502925530648558, 'ci95': [0.006029129881992304, 0.010642195878161084], 'unit': 'mechanism', 'units': 32}; error reduction {'mean': 3.8044578267607983e-07, 'ci95': [-2.541808665192137e-07, 1.101573874225103e-06], 'unit': 'mechanism', 'units': 32}.
- budget8-noise0.02-offset0.0-cost0.005: information gate B=True; finite-support C screen=False; raw gap recovery=0.000; adaptive-minus-fixed return {'mean': 0.003870312499999994, 'ci95': [0.00292472749255952, 0.004825266927083326], 'unit': 'mechanism', 'units': 32}; error reduction {'mean': 0.0, 'ci95': [0.0, 0.0], 'unit': 'mechanism', 'units': 32}.
- budget8-noise0.02-offset0.0-cost0.02: information gate B=True; finite-support C screen=False; raw gap recovery=0.000; adaptive-minus-fixed return {'mean': 0.015804315476190475, 'ci95': [0.012020345982142855, 0.019639181547619046], 'unit': 'mechanism', 'units': 32}; error reduction {'mean': 0.0, 'ci95': [0.0, 0.0], 'unit': 'mechanism', 'units': 32}.
- budget16-noise0.02-offset0.0-cost0.01: information gate B=True; finite-support C screen=False; raw gap recovery=0.000; adaptive-minus-fixed return {'mean': 0.007892485119047617, 'ci95': [0.005999808407738093, 0.009809990699404757], 'unit': 'mechanism', 'units': 32}; error reduction {'mean': 0.0, 'ci95': [0.0, 0.0], 'unit': 'mechanism', 'units': 32}.
- budget16-noise0.08-offset0.0-cost0.01: information gate B=True; finite-support C screen=True; raw gap recovery=0.153; adaptive-minus-fixed return {'mean': 0.07437327394401191, 'ci95': [0.03234755184941228, 0.12791801796250793], 'unit': 'mechanism', 'units': 32}; error reduction {'mean': 0.0007754853436067858, 'ci95': [0.00034887382678277894, 0.0013136697324114038], 'unit': 'mechanism', 'units': 32}.
- budget16-noise10.0-offset0.0-cost0.01: information gate B=False; finite-support C screen=False; raw gap recovery=0.000; adaptive-minus-fixed return {'mean': 0.0, 'ci95': [0.0, 0.0], 'unit': 'mechanism', 'units': 32}; error reduction {'mean': 0.0, 'ci95': [0.0, 0.0], 'unit': 'mechanism', 'units': 32}.
- budget16-noise0.02-offset0.03-cost0.01: information gate B=True; finite-support C screen=False; raw gap recovery=0.000; adaptive-minus-fixed return {'mean': 0.008502925530648558, 'ci95': [0.006029129881992304, 0.010642195878161084], 'unit': 'mechanism', 'units': 32}; error reduction {'mean': 3.8044578267607983e-07, 'ci95': [-2.541808665192137e-07, 1.101573874225103e-06], 'unit': 'mechanism', 'units': 32}.
- budget16-noise0.02-offset0.0-cost0.005: information gate B=True; finite-support C screen=False; raw gap recovery=0.000; adaptive-minus-fixed return {'mean': 0.003870312499999994, 'ci95': [0.00292472749255952, 0.004825266927083326], 'unit': 'mechanism', 'units': 32}; error reduction {'mean': 0.0, 'ci95': [0.0, 0.0], 'unit': 'mechanism', 'units': 32}.
- budget16-noise0.02-offset0.0-cost0.02: information gate B=True; finite-support C screen=False; raw gap recovery=0.000; adaptive-minus-fixed return {'mean': 0.015804315476190475, 'ci95': [0.012020345982142855, 0.019639181547619046], 'unit': 'mechanism', 'units': 32}; error reduction {'mean': 0.0, 'ci95': [0.0, 0.0], 'unit': 'mechanism', 'units': 32}.

## Interpretation and limits

The no-experiment method is prior-informed. Fixed, random, EIG and VOI use the same final posterior estimator. Additive identification changes the estimator and is shown separately. All methods share setup cost3 outside variable budget. The private nominal reference has privileged mechanism access and zero variable cost; it is not an attainable baseline.

The finite prior is explicitly conditioned on the same nominal admission rule used for tasks. Nonzero culture variation uses an approximate Gaussian likelihood, not an exact posterior. A failed one-step planner is not evidence against all adaptive experimental design.

Imitation/PPO execution checks and complete CPU/MPS benchmarks are recorded separately. Tiny smoke results cannot establish useful learned scientific behavior, PPO added value, or multiseed significance. Larger scientific training is withheld until development gates justify it.

Strong fixed schedules are preserved, including full-dose outcome identification. A documented development revision must address a demonstrated scientific limitation without weakening these baselines. Continuous-parameter, topology and range holdouts remain future gates; no locked-test outcome may guide revisions.

Total feasibility wall time: 193.92s. Peak process RSS: 0.292 GiB.
Code content hash: c0817970201735debdb58bc14e7ea88863cc4b8bab2ff88d6964e4d901616e37
Specification SHA256: f4afb36001041f62ba52a51026f0b48878474c2b25cce58027aee46c8ec2676a
