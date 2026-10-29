import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from rewards.schema import DIMENSIONS, SCORE_FIELDS, FIDELITY_GATES
from rewards.scoring import ContextualImageScorer
from rewards.score import score_record
from weight_head.model import M8WeightHead, save_checkpoint, load_checkpoint, context_indices
from weight_head.data import product_macro_mean, validate_splits, load_pairs
from weight_head.train import train

@pytest.fixture
def head_path(tmp_path):
    torch.manual_seed(42)
    model = M8WeightHead(3, 3)
    with torch.no_grad():
        model.global_weights.copy_(torch.linspace(.1, 1.1, 11))
        model.residual.weight.normal_(0, .1)
    vocab = {'market': {'AA': 1, 'BB': 2}, 'category_l1': {'Home': 1, 'Fashion': 2}}
    path = tmp_path / 'head.pt'
    save_checkpoint(path, model, vocab)
    return path


def judge():
    return {**{k: 'pass' for k in FIDELITY_GATES}, **dict(zip(SCORE_FIELDS, range(11)))}


def test_training_and_evaluation_share_exact_score(head_path):
    from rewards.strategy import DecoupledCotImageRewardStrategy
    config = {'weight_head': {'checkpoint': str(head_path)}}
    strategy = DecoupledCotImageRewardStrategy(config)
    context = {'market': 'AA', 'category_l1': 'Home'}
    payload = judge()
    reward = strategy._image_reward(payload, context)
    evaluated = score_record(ContextualImageScorer(head_path), {**context, 'judge': judge()})
    assert reward == evaluated['reward'] == evaluated['overall'] / 10
    assert payload['contextual_scoring']['weights'] == evaluated['weights']
    expected = sum(a * b for a, b in zip(evaluated['weights'], range(11))) / sum(evaluated['weights'])
    assert evaluated['overall'] == pytest.approx(expected)
    assert not strategy.image_scorer.model.training
    assert all(not p.requires_grad for p in strategy.image_scorer.model.parameters())


def test_context_varies_weights_but_candidates_share_them(head_path):
    scorer = ContextualImageScorer(head_path)
    assert scorer.weights('AA', 'Home') == scorer.weights('aa', 'Home')
    assert scorer.weights('AA', 'Home') != scorer.weights('BB', 'Home')
    assert scorer.weights('AA', 'Home') != scorer.weights('AA', 'Fashion')
    a = scorer.score(judge(), {'market': 'AA', 'category_l1': 'Home'})
    p = judge();p[SCORE_FIELDS[1]] = 9
    b = scorer.score(p, {'market': 'AA', 'category_l1': 'Home'})
    assert a['weights'] == b['weights']


def test_pair_order_symmetry_and_11_outputs(head_path):
    model, _, _ = load_checkpoint(head_path)
    a,b = torch.rand(3,11)*10,torch.rand(3,11)*10
    mi,ci = torch.tensor([1,2,1]),torch.tensor([1,1,2])
    ab,w=model(a,b,mi,ci);ba,_=model(b,a,mi,ci)
    assert w.shape==(3,11)
    torch.testing.assert_close(ab,-ba)
    torch.testing.assert_close(ab.sigmoid()+ba.sigmoid(),torch.ones(3))


@pytest.mark.parametrize('field', ['dimensions','score_scale','architecture'])
def test_reject_incompatible_checkpoint(head_path, tmp_path, field):
    data = torch.load(head_path, weights_only=True)
    data[field] = list(DIMENSIONS[:8]) if field=='dimensions' else None
    path=tmp_path/'old.pt';torch.save(data,path)
    with pytest.raises(ValueError, match='Incompatible'):
        load_checkpoint(path)


def test_reject_permuted_dimension_order(head_path, tmp_path):
    data=torch.load(head_path,weights_only=True);data['dimensions'].reverse()
    path=tmp_path/'permuted.pt';torch.save(data,path)
    with pytest.raises(ValueError):load_checkpoint(path)


def test_missing_context_unknown_and_bad_weight_sum(head_path, tmp_path):
    scorer=ContextualImageScorer(head_path)
    with pytest.raises(ValueError):scorer.score(judge(),{'market':'AA'})
    with pytest.raises(ValueError):scorer.weights('CC','Home')
    assert len(ContextualImageScorer(head_path,unknown_context='zero_embedding').weights('CC','Home'))==11
    model,vocab,_=load_checkpoint(head_path)
    with torch.no_grad():
        for p in model.parameters():p.zero_()
    path=tmp_path/'zero.pt';save_checkpoint(path,model,vocab)
    with pytest.raises(ValueError,match='Weight sum'):ContextualImageScorer(path).weights('AA','Home')


def test_fidelity_fail_zero_not_a_missing_observation(head_path):
    scorer=ContextualImageScorer(head_path);p={k:'pass' for k in FIDELITY_GATES};p[FIDELITY_GATES[0]]='fail'
    assert scorer.score(p,{'market':'AA','category_l1':'Home'})['reward']==0
    p.pop(FIDELITY_GATES[0])
    with pytest.raises(ValueError,match='gates'):scorer.score(p,{'market':'AA','category_l1':'Home'})
    p=judge();p[SCORE_FIELDS[0]]=float('nan')
    with pytest.raises(ValueError):scorer.score(p,{'market':'AA','category_l1':'Home'})


def test_macro_weighting_is_not_pair_mean():
    b={'products':2,'product_index':torch.tensor([0,0,1]),'sample_weight':torch.tensor([1.,3.,1.])}
    assert product_macro_mean(torch.tensor([0.,4.,1.]),b).item()==2.0


def make_pair(product):
    return {'product_id':product,'market':'AA','category_l1':'Home',
            'scores_a':dict(zip(DIMENSIONS,[8.]*11)), 'scores_b':dict(zip(DIMENSIONS,[3.]*11)),
            'target':.9,'sample_weight':1.}


def test_two_stage_cpu_training_and_reload(tmp_path):
    data={}
    for split in ['pretrain','finetune','pretrain_validation','finetune_validation']:
        p=tmp_path/(split+'.jsonl')
        p.write_text('\n'.join(json.dumps(make_pair(split+str(i))) for i in range(3))+'\n')
        data[split]=str(p)
    recipe={'steps':4,'learning_rate':.03,'global_l2':.01,'residual_l2':.1,'validation_every':2,'patience':4}
    output=train({'_project_dir':str(tmp_path),'data':data,'seed':42,'output_dir':'out','pretrain':recipe,'finetune':recipe})
    model,vocab,metadata=load_checkpoint(output)
    assert metadata['stage']=='finetune' and metadata['pretrain']['selected_step']>0
    assert metadata['finetune']['selected_step']>0
    assert (output.parent/'pretrain.pt').exists()
    scorer=ContextualImageScorer(output)
    assert 0<=scorer.score(judge(),{'market':'AA','category_l1':'Home'})['reward']<=1
    model_pre,_,_=load_checkpoint(output.parent/'pretrain.pt')
    assert not torch.equal(model.global_weights,model_pre.global_weights)


def test_training_validation_overlap_rejected():
    r=make_pair('same')
    with pytest.raises(ValueError,match='overlap'):validate_splits([r],[],[],[r])


def test_actual_strategy_context_propagation(head_path):
    from rewards.strategy import DecoupledCotImageRewardStrategy
    from tests.test_element_kernel_advantage import completion
    strategy=DecoupledCotImageRewardStrategy({'weight_head':{'checkpoint':str(head_path)},
        'online_reward':{'enable_cot_reward':False}})
    request={'request_id':'test','row_key':'product','sample_index':0,'raw_completion':completion('Test'),
             'schema_valid':True,'market':'AA','category_l1':'Home'}
    with patch.object(strategy,'_generate_image',side_effect=lambda r:{'_request':r}), \
         patch.object(strategy,'_judge_image_from_generated',return_value=judge()):
        rows=strategy.score([request])
    assert rows[0].active
    assert rows[0].image_reward==strategy.image_scorer.score(judge(),request)['reward']


def test_image_judge_entry_uses_same_weighting(head_path):
    from rewards.strategy import DecoupledCotImageRewardStrategy
    from rewards.judge_images import judge_record
    strategy=DecoupledCotImageRewardStrategy({'weight_head':{'checkpoint':str(head_path)}})
    row={'market':'AA','category_l1':'Home','original_image_url':'source.png','generated_image_url':'edit.png'}
    with patch.object(strategy,'_judge_image',return_value=judge()) as mock:
        out=judge_record(strategy,row,3)
    assert mock.call_count==3
    assert out['overall']==pytest.approx(strategy.image_scorer.score(judge(),row)['overall'])
    assert all(s['weights']==out['judgments'][0]['weights'] for s in out['judgments'])


def test_local_source_image_is_inlined_for_cot(head_path,tmp_path):
    from PIL import Image
    from rewards.strategy import DecoupledCotImageRewardStrategy
    p=tmp_path/'source.png';Image.new('RGB',(8,8),'white').save(p)
    strategy=DecoupledCotImageRewardStrategy({'weight_head':{'checkpoint':str(head_path)}})
    part=strategy._chat_image_part(str(p),model='gemini-2.5-flash')
    assert part['image_url']['url'].startswith('data:image/png;base64,')


def test_dataset_carries_l1_to_reward_request(tmp_path):
    import csv
    from PIL import Image
    from utils.data import build_dataset
    image=tmp_path/'source.png';Image.new('RGB',(8,8)).save(image)
    data=tmp_path/'train.csv'
    row={'original_image_url':str(image),'title':'test','market':'AA','category_l1':'Home'}
    with data.open('w') as h:
        writer=csv.DictWriter(h,fieldnames=list(row));writer.writeheader();writer.writerow(row)
    dataset=build_dataset({'_project_dir':str(tmp_path),'weight_head':{'checkpoint':'unused'},
        'data':{'train_csv':str(data),'user_template':'{title}','market_column':'market',
                'category_l1_column':'category_l1'}},None)
    assert dataset[0]['market']=='AA' and dataset[0]['category_l1']=='Home'
    row.pop('category_l1')
    with data.open('w') as h:
        writer=csv.DictWriter(h,fieldnames=list(row));writer.writeheader();writer.writerow(row)
    with pytest.raises(ValueError,match='category_l1'):
        build_dataset({'_project_dir':str(tmp_path),'weight_head':{'checkpoint':'unused'},
                      'data':{'train_csv':str(data),'user_template':'{title}'}},None)


def test_inline_media_is_not_copied_to_public_audit(head_path,tmp_path):
    from PIL import Image
    from rewards.strategy import DecoupledCotImageRewardStrategy,image_data_url_from_file
    image=tmp_path/'source.png';Image.new('RGB',(8,8)).save(image)
    strategy=DecoupledCotImageRewardStrategy({'weight_head':{'checkpoint':str(head_path)}})
    data=image_data_url_from_file(image)
    refs=strategy._generated_image_refs(data,{'request_id':'unit'})
    assert refs['_generated_ref_for_judge']==data
    assert all('base64,' not in str(v) for k,v in refs.items() if not k.startswith('_'))
