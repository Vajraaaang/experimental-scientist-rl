import hashlib
import numpy as np
import pytest
import torch
from experimentlab.baselines import fixed_candidates
from experimentlab.envs import PilotConfig
from experimentlab.tasks import finite_family
from experimentlab.inference import PredictionBank
from experimentlab.evaluation import tune_fixed,development_tasks
from experimentlab.learning import ActorCritic,save_checkpoint
from experimentlab.experiments import evaluate_checkpoint


def test_fixed_design_tuner_respects_supported_smaller_caps():
    assert max(map(len,fixed_candidates(16,max_experiments=1)))<=1
    bank=PredictionBank(finite_family(),PilotConfig(budget=16,max_experiments=1))
    tasks,_=development_tasks(bank,2,721)
    schedule,_=tune_fixed(bank,tasks,beam_width=1)
    assert len(schedule)<=1


def test_checkpoint_binds_explicit_actual_manifest(tmp_path):
    manifest=tmp_path.resolve()/'actual-manifest.json'
    manifest.write_text('{"training_pool":7401,"validation_pool":8401}')
    model=ActorCritic();optimizer=torch.optim.Adam(model.parameters())
    checkpoint=tmp_path.resolve()/'model.pt'
    save_checkpoint(checkpoint,model,optimizer,{},provenance={'manifest_path':str(manifest)})
    state=torch.load(checkpoint,weights_only=False)
    assert state['provenance']['manifest_hash']==hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_generic_evaluator_cannot_silently_use_wrong_research_prior(tmp_path):
    model=ActorCritic();optimizer=torch.optim.Adam(model.parameters())
    checkpoint=tmp_path.resolve()/'model.pt'
    save_checkpoint(checkpoint,model,optimizer,{},provenance={'scope':'development_multiseed','prior':'unfiltered'})
    with pytest.raises(ValueError,match='research evaluation'):
        evaluate_checkpoint(checkpoint)
