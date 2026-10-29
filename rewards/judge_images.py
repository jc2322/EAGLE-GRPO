"""Judge supplied images and aggregate with the frozen training head; no benchmark bundled."""
import argparse
import json
import math
from pathlib import Path
from utils.config import load_config
from rewards.strategy import DecoupledCotImageRewardStrategy


def judge_record(strategy, record, repeats):
    if repeats < 1:
        raise ValueError('repeats must be positive')
    strategy.image_scorer.weights(record.get('market'), record.get('category_l1'))
    required = ('original_image_url', 'generated_image_url')
    if any(not record.get(key) for key in required):
        raise ValueError('Source and generated image references are required')
    scores = []
    for i in range(repeats):
        request = dict(record)
        request['request_id'] = str(record.get('request_id', 'offline')) + f'-repeat-{i}'
        payload, _ = strategy._with_retry('judge', lambda: strategy._judge_image(
            request, str(record.get('compiled_prompt', '')), record['generated_image_url']))
        scores.append(strategy.image_scorer.score(payload, request))
    return {'item_id': record.get('item_id'), 'market': record['market'],
            'category_l1': record['category_l1'], 'judgments': scores,
            'overall': math.fsum(s['overall'] for s in scores) / repeats,
            'reward': math.fsum(s['reward'] for s in scores) / repeats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/grpo_gpt.yaml')
    parser.add_argument('--input', required=True, help='JSONL: product metadata, source and candidate paths/URLs')
    parser.add_argument('--output', required=True)
    parser.add_argument('--judge-model', help='Override image judge; configure its endpoint/key via JUDGE_* env vars')
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1 or Path(args.input).resolve() == Path(args.output).resolve():
        parser.error('Use a positive repeat count and a separate output path')
    config = load_config(args.config)
    if args.judge_model:
        config['online_reward']['image_judge']['model'] = args.judge_model
    strategy = DecoupledCotImageRewardStrategy(config)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.input) as source, open(args.output, 'w') as destination:
        for index, line in enumerate(source):
            if line.strip():
                record = json.loads(line)
                record['request_id'] = f'offline-{index}'
                destination.write(json.dumps(judge_record(strategy, record, args.repeats)) + '\n')

if __name__ == '__main__':
    main()
