"""Original/optimized pretraining followed by optimized/optimized fine-tuning."""
import argparse
import copy
import logging
import random
from pathlib import Path
import torch
from torch.nn import functional as F
from utils.config import load_config, resolve_path
from weight_head.data import load_pairs, validate_splits, tensor_batch, product_macro_mean
from weight_head.model import M8WeightHead, build_vocab, save_checkpoint

LOG = logging.getLogger(__name__)


def objective(model, batch, *, global_l2, residual_l2):
    logits, coefficients = model(batch['scores_a'], batch['scores_b'], batch['markets'], batch['categories'])
    bce = F.binary_cross_entropy_with_logits(logits, batch['target'], reduction='none')
    data_loss = product_macro_mean(bce, batch)
    residual = coefficients - model.global_weights
    penalty = global_l2 * model.global_weights.square().mean()
    penalty = penalty + residual_l2 * product_macro_mean(residual.square().mean(-1), batch)
    return data_loss + penalty, data_loss


def fit_stage(model, train, validation, recipe):
    steps = int(recipe['steps'])
    every = int(recipe.get('validation_every', 10))
    patience = int(recipe.get('patience', 80))
    lr = float(recipe['learning_rate'])
    regularization = {name: float(recipe.get(name, 0.0)) for name in ('global_l2', 'residual_l2')}
    if steps < 1 or every < 1 or patience < 1 or lr <= 0 or min(regularization.values()) < 0:
        raise ValueError('Invalid training recipe')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    best_state, best_loss, selected_step = None, float('inf'), 0
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = objective(model, train, **regularization)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe.get('max_grad_norm', 5.0)))
        optimizer.step()
        if step % every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                _, validation_loss = objective(model, validation, **regularization)
            value = float(validation_loss)
            if not torch.isfinite(validation_loss):
                raise ValueError('Nonfinite validation loss')
            LOG.info('step=%d train_loss=%.6f validation_bce=%.6f', step, float(loss.detach()), value)
            if value < best_loss - float(recipe.get('min_delta', 1e-5)):
                best_state = copy.deepcopy(model.state_dict())
                best_loss, selected_step = value, step
            if step - selected_step >= patience:
                break
    model.load_state_dict(best_state)
    return {'selected_step': selected_step, 'validation_bce': best_loss, 'recipe': dict(recipe)}


def train(config):
    root = Path(config['_project_dir'])
    data = {name: load_pairs(resolve_path(config['data'][name], base_dir=root)) for name in
            ('pretrain', 'finetune', 'pretrain_validation', 'finetune_validation')}
    validate_splits(**data)
    # Context identities from both training stages only; no validation/test fitting.
    vocab = build_vocab(data['pretrain'] + data['finetune'])
    seed = int(config.get('seed', 42))
    random.seed(seed); torch.manual_seed(seed)
    device = torch.device(config.get('device', 'cpu'))
    model = M8WeightHead(len(vocab['market']) + 1, len(vocab['category_l1']) + 1).to(device)
    batches = {name: tensor_batch(rows, vocab, device, unknown=config.get('unknown_context', 'error'))
               for name, rows in data.items()}
    output = resolve_path(config.get('output_dir', './outputs/weight_head'), base_dir=root)
    metadata = {'seed': seed, 'training_products': {name: batch['products'] for name, batch in batches.items()},
                'recipe': 'V2-M8 adapted to 11 quality dimensions; new training required'}
    for stage in ('pretrain', 'finetune'):
        result = fit_stage(model, batches[stage], batches[stage + '_validation'], config[stage])
        metadata[stage] = result
        metadata['stage'] = stage
        save_checkpoint(output / (stage + '.pt'), model, vocab, metadata=copy.deepcopy(metadata))
        LOG.info('Saved %s checkpoint selected at step %d', stage, result['selected_step'])
    return output / 'finetune.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/weight_head.yaml')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    train(load_config(args.config))

if __name__ == '__main__':
    main()
