"""Create and summarize an independent, manual answer-review worksheet.

This does not call a model or judge legal correctness. Reviewers must inspect
actual answers and sources and fill in their own labels outside the repository.
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('id', 'question', 'expected_scope', 'observed_mode', 'response_reference',
          'correctness', 'citation_support', 'appropriate_abstention', 'reviewer', 'notes')
RATINGS = ('correctness', 'citation_support', 'appropriate_abstention')
ALLOWED = {'', 'yes', 'no', 'uncertain', 'not_applicable'}


def template(path):
    dataset = json.loads((ROOT / 'evaluation' / 'questions.json').read_text(encoding='utf-8'))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for case in dataset['cases']:
            writer.writerow({'id': case['id'], 'question': case['query'],
                             'expected_scope': 'historical' if 'section' in case else 'unsupported'})


def summarize(rows):
    ids = set()
    summary = {field: {value: 0 for value in ALLOWED if value} for field in RATINGS}
    reviewed = 0
    for row in rows:
        case_id = (row.get('id') or '').strip()
        if not case_id or case_id in ids:
            raise ValueError('Each review row must have a unique, non-empty case ID.')
        ids.add(case_id)
        values = {field: (row.get(field) or '').strip().lower() for field in RATINGS}
        for field, value in values.items():
            if value not in ALLOWED:
                raise ValueError(f'{case_id}: invalid {field} rating: {value}')
        if any(values.values()):
            if not (row.get('reviewer') or '').strip() or not (row.get('response_reference') or '').strip():
                raise ValueError(f'{case_id}: rated responses need a reviewer and response reference.')
            reviewed += 1
            for field, value in values.items():
                if value:
                    summary[field][value] += 1
    return {'rows': len(rows), 'reviewed': reviewed, 'unreviewed': len(rows) - reviewed,
            'ratings': summary, 'note': 'Manual ratings only; reviewer expertise and representative sampling must be documented separately.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--template', type=Path, help='Write a blank worksheet to this path')
    group.add_argument('--score', type=Path, help='Summarize a completed worksheet')
    args = parser.parse_args()
    if args.template:
        if args.template.exists():
            raise FileExistsError('Refusing to overwrite an existing review worksheet.')
        template(args.template)
        print(f'Blank worksheet written to {args.template}; no answers have been reviewed.')
    else:
        with args.score.open(encoding='utf-8-sig', newline='') as stream:
            print(json.dumps(summarize(list(csv.DictReader(stream))), indent=2))


if __name__ == '__main__':
    main()
