"""Score supplied judge outputs using the same frozen head as GRPO (no benchmark)."""
import argparse
import json
from pathlib import Path
from rewards.scoring import ContextualImageScorer

def score_record(scorer, record):
    result = scorer.score(record['judge'], record)
    return {'item_id': record.get('item_id'), **result}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True, help='JSONL with market, category_l1, and judge payload')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.input).resolve() == Path(args.output).resolve():
        parser.error('Output must differ from input')
    scorer = ContextualImageScorer(args.checkpoint)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.input) as source, open(args.output, 'w') as destination:
        for line in source:
            if line.strip():
                destination.write(json.dumps(score_record(scorer, json.loads(line))) + '\n')

if __name__ == '__main__':
    main()
