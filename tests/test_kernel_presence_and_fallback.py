import pytest
import torch
from tests.test_element_kernel_advantage import completion, padded_char_batch, CharTokenizer
from rewards.online_component_reward import ComponentReward, ComponentRewardBatch, ComponentRewardStore, normalize_component_advantages
from utils.advantage_protocol import AdvantageContext
from utils.element_kernel_advantage import ElementKernelAdvantageProvider


def run_group(*, eta=.5, partial_hidden=False, strict=False):
    texts=[completion('Short'),completion('Long headline'),completion('Compact')]
    texts[1]=texts[1].replace('<subtitle></subtitle>','<subtitle>Useful feature</subtitle>')
    rewards=[ComponentReward(request_id=str(i),row_key='p',sample_index=i,raw_completion=t,
             schema_valid=True,cot_reward=.3,image_reward=float(i),scalar_reward=float(i)) for i,t in enumerate(texts)]
    normalize_component_advantages(rewards,min_successful_samples=1,min_successful_ratio=0.)
    store=ComponentRewardStore();store.set_latest(ComponentRewardBatch(rewards))
    ids,mask=padded_char_batch(texts)
    cfg={'element_tags':['headline','subtitle'],'eta':eta,'lambda_reg':.1,'include_thinking':False,
         'condition_number_max':1e9,'embedding_backend':'hidden_states' if partial_hidden else 'hashed_ngram',
         'hidden_state_fallback':'error' if strict else 'hashed_ngram'}
    provider=ElementKernelAdvantageProvider(store,{'element_credit':cfg})
    hidden=torch.randn(2,ids.shape[1],8) if partial_hidden else None
    advantages=provider.token_advantages(AdvantageContext(ids,mask,torch.zeros(3),CharTokenizer(),hidden))
    return rewards,provider,advantages


@pytest.mark.parametrize('eta',[0.,.5,1.])
def test_optional_field_absence_preserves_mean(eta):
    rewards,provider,a=run_group(eta=eta)
    for i,r in enumerate(rewards):
        selected=provider.last_loss_mask[i].bool()
        assert a[i][selected].mean().item()==pytest.approx(r.image_advantage,abs=1e-6)
        if eta==0:
            torch.testing.assert_close(a[i][selected],torch.full_like(a[i][selected],r.image_advantage))


def test_partial_hidden_states_do_not_mix_feature_dimensions():
    rewards,provider,a=run_group(partial_hidden=True)
    assert any(row['fallback']=='group_field_hashed_representation' for row in provider.last_diagnostics)
    for i,r in enumerate(rewards):
        selected=provider.last_loss_mask[i].bool()
        assert a[i][selected].mean().item()==pytest.approx(r.image_advantage,abs=1e-6)


def test_strict_hidden_state_mode_rejects_partial_groups():
    with pytest.raises(ValueError,match='hidden-state'):
        run_group(partial_hidden=True,strict=True)
