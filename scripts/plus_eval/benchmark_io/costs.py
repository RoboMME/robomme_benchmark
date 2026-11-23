"""Provider usage and list-price estimates, never inferred from transcript length.

Bedrock US geographic cross-region rates, effective 2026-11-09. Estimates
exclude account discounts/taxes; incomplete responses remain explicitly unknown.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

RATE_CARD = {
    'model': 'us.anthropic.claude-opus-5-5',
    'provider': 'AWS Bedrock', 'scope': 'US geographic cross-region, standard',
    'effective_date': '2026-11-09', 'currency': 'USD',
    'source': 'https://assets.claude.com/a04aca860a54b7323e33f449706e9b43680e1346.pdf',
    'source_page': 6,
    'usd_per_million_tokens': {
        'input_tokens': 4.40, 'output_tokens': 22.00,
        'cache_read_input_tokens': .22,
        'ephemeral_5m_input_tokens': 5.50, 'ephemeral_1h_input_tokens': 8.80,
    },
}


def estimate(usage):
    """Do not double-count cache totals and their duration breakdown."""
    counts = {k: int(usage.get(k) or 0) for k in
              ('input_tokens', 'output_tokens', 'cache_read_input_tokens')}
    creation = usage.get('cache_creation') or {}
    for key in ('ephemeral_5m_input_tokens', 'ephemeral_1h_input_tokens'):
        counts[key] = int(creation.get(key) or 0)
    unresolved = int(usage.get('cache_creation_input_tokens') or 0) - sum(creation.get(k, 0) or 0 for k in
                  ('ephemeral_5m_input_tokens', 'ephemeral_1h_input_tokens'))
    if unresolved < 0:
        raise ValueError('Provider cache breakdown exceeds total cache creation')
    rates = RATE_CARD['usd_per_million_tokens']
    known = sum(counts[k] * rates[k] for k in rates) / 1_000_000
    return {'token_counts': counts, 'unresolved_cache_creation_tokens': unresolved,
            'estimated_cost_usd': round(known, 10) if not unresolved else None,
            'estimated_cost_range_usd': [round(known + unresolved * rate / 1_000_000, 10)
                                         for rate in (5.50, 8.80)],
            'basis': 'provider-reported usage × published list prices; not an invoice'}


def record_usage(record, usage, *, complete, context=None):
    record = Path(record)
    (record / 'usage.json').write_text(json.dumps(usage, indent=2))
    report = {**estimate(usage), 'usage_complete': complete, 'context': context or {},
              'rate_card': RATE_CARD}
    if not complete:
        # A partially streamed output is a lower bound, never a zero-cost call.
        report['estimated_cost_usd'] = None
    (record / 'cost.json').write_text(json.dumps(report, indent=2))
    return report


def summarize(root):
    root = Path(root)
    groups = {}
    records = {p.parent for p in root.rglob('cost.json')}
    for path in root.rglob('transport.json'):
        transport = json.loads(path.read_text())
        if transport.get('model') == RATE_CARD['model']:
            records.add(path.parent)
    for record in sorted(records):
        path = record / 'cost.json'
        pending = not path.is_file()
        if pending:
            transport = json.loads((record / 'transport.json').read_text())
            row = {**estimate({}), 'usage_complete': False, 'context': transport.get('context', {})}
        else:
            row = json.loads(path.read_text())
        context = row.get('context', {})
        key = tuple(context.get(k, 'unspecified') for k in ('method', 'setting', 'budget', 'phase'))
        group = groups.setdefault(key, {'method': key[0], 'setting': key[1], 'budget': key[2],
            'phase': key[3], 'requests': 0, 'incomplete_requests': 0, 'pending_usage_requests': 0,
            'known_estimated_cost_usd': 0., 'unresolved_cache_creation_tokens': 0, 'token_counts': Counter()})
        group['requests'] += 1
        group['incomplete_requests'] += not row['usage_complete']
        group['pending_usage_requests'] += pending
        group['known_estimated_cost_usd'] += row['estimated_cost_range_usd'][0]
        group['unresolved_cache_creation_tokens'] += row['unresolved_cache_creation_tokens']
        group['token_counts'].update(row['token_counts'])
    return {'rate_card': RATE_CARD, 'groups': list(groups.values())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = json.dumps(summarize(args.root), indent=2)
    if args.output:
        args.output.write_text(result)
    else:
        print(result)


if __name__ == '__main__':
    main()
