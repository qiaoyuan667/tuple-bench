"""Explicit view-only features. Neither fitting API accepts full cases or raw runs."""
from __future__ import annotations
import re
from collections import Counter
from .io import require, finite_number
PAYLOAD_KEYS = {'choice', 'confidence', 'max_probability', 'margin', 'top2_ids', 'top2', 'probabilities', 'execution_status'}
TRACE_KEYS = {'initial_action', 'chosen_check_id', 'check_answer', 'final_route', 'execution_status'}

def public_features(consumer_view: dict) -> dict[str, float]:
    require(set(consumer_view) == {'recipient_role', 'public', 'public_task', 'downstream_contexts'}, 'Unexpected consumer-view fields')
    public = consumer_view['public']
    require(set(public) == {'request', 'intake_channel', 'operating_window', 'case_context'}, 'Public schema is explicit; update builder for new declared fields')
    f = {f"role={consumer_view['recipient_role']}": 1.0}
    for field, text in public.items():
        require(isinstance(text, str), 'Public values must be declared strings')
        if field in ('request', 'case_context'):
            for token, n in Counter(re.findall('[a-z0-9]+', text.lower())).items():
                f[f'public:{field}:token={token}'] = float(n)
        else:
            f[f'public:{field}={text}'] = 1.0
    for i, c in enumerate(consumer_view['public_task']['candidates']):
        f[f"candidate_position:{i}={c['id']}"] = 1.0
    return f

def payload_features(payload: dict) -> dict[str, float]:
    require(set(payload) <= PAYLOAD_KEYS, 'Undeclared released field')
    f = {}
    for k in ('choice', 'execution_status'):
        if k in payload:
            f[f'release:{k}={payload[k]}'] = 1.0
    for k in ('confidence', 'max_probability', 'margin'):
        if k in payload:
            require(finite_number(payload[k]), 'Invalid scalar release')
            f['release:' + k] = float(payload[k])
            f['observed:' + k] = 1.0
    if 'probabilities' in payload:
        for k, v in payload['probabilities'].items():
            f['release:p:' + k] = float(v)
            f['observed:p:' + k] = 1.0
    for i, k in enumerate(payload.get('top2_ids', [])):
        f[f'release:rank:{i}={k}'] = 1.0
    for i, row in enumerate(payload.get('top2', [])):
        require(set(row) == {'id', 'score'}, 'Unexpected top2 field')
        k = row['id']
        f[f'release:rank:{i}={k}'] = 1.0
        f['release:topscore:' + k] = float(row['score'])
        f['observed:topscore:' + k] = 1.0
    return f

def utility_features(consumer_view: dict, payload: dict) -> dict[str, float]:
    return {**public_features(consumer_view), **payload_features(payload)}

def privacy_features(consumer_view: dict, payload: dict, T: str, hypotheses: list[str], trace: dict | None=None, variant: str='linear') -> dict[str, float]:
    require(len(hypotheses) == 2, 'Binary closed-set audit expects two hypotheses')
    f = utility_features(consumer_view, payload)
    f[f'allowed_T={T}'] = 1.0
    for i, name in enumerate(hypotheses):
        f[f'hypothesis:{i}={name}'] = 1.0
    if trace is not None:
        require(set(trace) <= TRACE_KEYS, 'Undeclared transcript field')
        for k, v in trace.items():
            if v is not None:
                f[f'trace:{k}={v}'] = 1.0
    if variant == 'quadratic':
        nums = {k: v for k, v in f.items() if k.startswith(('release:p:', 'release:topscore:')) or k in ['release:confidence', 'release:max_probability', 'release:margin']}
        for k, v in nums.items():
            f['square:' + k] = v * v
        keys = sorted(nums)
        for i, k in enumerate(keys):
            for h in keys[i + 1:]:
                f['product:' + k + '*' + h] = nums[k] * nums[h]
    else:
        require(variant == 'linear', 'Unknown privacy feature family')
    base = list(f.items())
    condition = T + '|' + ','.join(hypotheses)
    for k, v in base:
        if k.startswith(('release:', 'observed:', 'trace:', 'square:', 'product:')):
            f['condition:' + condition + '|' + k] = v
    return f
