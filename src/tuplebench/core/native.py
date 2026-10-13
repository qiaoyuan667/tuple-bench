"""Native-run validation/import and opt-in, serial, budget-bounded HTTP execution."""
from __future__ import annotations
import copy
import json
import os
import random
import time
import urllib.request
import urllib.error
from pathlib import Path
from collections import Counter
from jsonschema import Draft202012Validator
from .io import require, read_jsonl, write_json, write_jsonl, append_jsonl, digest, file_digest, new_directory, parse_json, ContractError
from .releases import validate_response, order_from_case
from .schemas import RUN_SCHEMA

def to_request(case: dict, model: str) -> dict:
    mv = case['model_view']
    q = mv['question']
    return {'model': model, 'state': {'recipient_role': mv['recipient_role'], 'permission': copy.deepcopy(mv['permission']), **copy.deepcopy(mv['state'])}, 'questions': {q['id']: {'type': 'choice', 'instructions': q['instructions'], 'criteria': {c['id']: c['description'] for c in q['candidates']}}}}

def canonical_response(case: dict, raw: dict) -> tuple[str, dict]:
    require(isinstance(raw.get('model'), str) and raw['model'], 'Raw response has no actual model identity')
    qid = case['model_view']['question']['id']
    answer = raw.get('answers', {}).get(qid)
    require(isinstance(answer, dict), 'Missing requested answer')
    require(answer.get('type') == 'choice', 'Answer type does not match Choice')
    response = {k: answer[k] for k in ['choice', 'probabilities', 'confidence'] if k in answer}
    validate_response(response, order_from_case(case))
    return (raw['model'], response)

def load_runs(ds, path, allow_incomplete: bool=False, allow_synthetic: bool=False):
    rows = read_jsonl(path)
    runs = {}
    issues = []
    validator = Draft202012Validator(RUN_SCHEMA)
    versions = set()
    origins = set()
    for r in rows:
        errors = list(validator.iter_errors(r))
        require(not errors, 'Run schema: ' + '; '.join((e.message for e in errors[:3])))
        cid = r['case_id']
        require(cid in ds.case_map, 'Unknown case_id in runs; v1 IDs cannot be silently remapped')
        require(cid not in runs, 'Duplicate run for a case; use a separate repeat file/track')
        c = ds.case_map[cid]
        require(r['model_view_sha256'] == c['evaluator_only']['model_view_sha256'], 'Run input hash mismatch')
        origins.add(r['origin'])
        if r['status'] == 'completed':
            require(isinstance(r['model_version'], str) and r['model_version'], 'Actual model version required')
            versions.add(r['model_version'])
            issues.extend(({'case_id': cid, 'warning': w} for w in validate_response(r['response'], order_from_case(c), ds.engine['probability_sum_tolerance'])))
        else:
            require(r['response'] is None, 'Failed/pending responses must be null')
        runs[cid] = r
    missing = set(ds.case_map) - set(runs)
    require(not missing, 'Run file must contain one status row for every case, including not_executed')
    require(len(versions) <= 1, 'Mixed model versions: analyze separately')
    require(not ('synthetic_test' in origins and len(origins) > 1), 'Do not mix synthetic and native observations')
    require(allow_synthetic or 'synthetic_test' not in origins, 'Synthetic controls require explicit --allow-synthetic')
    pending = sum((r['status'] == 'not_executed' for r in runs.values()))
    require(allow_incomplete or pending == 0, 'Incomplete run; complete it or use explicit development --allow-incomplete')
    return (runs, {'counts': dict(Counter((r['status'] for r in rows))), 'origins': sorted(origins), 'model_versions': sorted(versions), 'input_run_sha256': file_digest(path), 'warnings': issues, 'incomplete_as_fallback': bool(pending), 'fallback': 'generic unavailable status + trusted review at cost 2; no detailed errors or timing released'})
