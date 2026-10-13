"""Load frozen cases, verify certificates and declared split lineages."""
from __future__ import annotations
import copy
import itertools
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from jsonschema import Draft202012Validator
from .io import read_json, read_jsonl, write_json, write_jsonl, require, digest, opaque, new_directory, file_digest, ContractError
from .contracts import class_ids, check_law, certify_pair, validate_specs
from .schemas import CASE_SCHEMA, REVIEW_SCHEMA, RESEARCH_REVIEW_CHECKS, export_schemas

@dataclass
class Dataset:
    path: Path
    manifest: dict
    domains: dict
    consumers: dict
    releases: list[dict]
    engine: dict
    cases: list[dict]
    bases: list[dict]
    certificates: list[dict]
    reviews: list[dict]

    @property
    def case_map(self):
        return {c['case_id']: c for c in self.cases}

    @property
    def fingerprint(self):
        return self.manifest['dataset_sha256']

def load_dataset(path: str | Path, review_path: str | Path | None=None) -> Dataset:
    p = Path(path)
    m = read_json(p / 'manifest.json')
    for name, h in m['artifact_sha256'].items():
        require(file_digest(p / name) == h, f'Frozen artifact changed: {name}; create a new dataset version')
    return Dataset(p, m, {x.stem: read_json(x) for x in sorted((p / 'domain_specs').glob('*.json'))}, {x.stem: read_json(x) for x in sorted((p / 'consumer_specs').glob('*.json'))}, read_json(p / 'release_specs.json'), read_json(p / 'engine.json'), read_jsonl(p / 'cases.jsonl'), read_jsonl(p / 'bases.jsonl'), read_jsonl(p / 'certificates/equivalence.jsonl'), read_jsonl(review_path or p / 'human_review/review_template.jsonl'))

def validate_dataset(ds: Dataset, require_reviewed: bool=False, review_policy: str='independent_human', judge_consensus=None) -> dict:
    require(review_policy in {'independent_human', 'llm_consensus'}, 'Unknown semantic review policy')
    require(judge_consensus is None or review_policy == 'llm_consensus', 'Choose llm_consensus explicitly when supplying judge evidence')
    validate_specs(ds.domains, ds.consumers, ds.releases)
    cases = ds.cases
    require(len(ds.case_map) == len(cases), 'Duplicate case_id')
    require(len({b['base_id'] for b in ds.bases}) == len(ds.bases), 'Duplicate base_id')
    groups = defaultdict(list)
    lineages = defaultdict(set)
    renderings = defaultdict(set)
    certs = {r['base_id']: r for r in ds.certificates}
    reviews = {r['base_id']: r for r in ds.reviews}
    require(len(certs) == len(ds.certificates), 'Duplicate certificate')
    require(len(reviews) == len(ds.reviews), 'Duplicate review')
    schema = Draft202012Validator(CASE_SCHEMA)
    for c in cases:
        errors = list(schema.iter_errors(c))
        require(not errors, 'Case schema: ' + '; '.join((e.message for e in errors[:3])))
        e = c['evaluator_only']
        d = ds.domains[c['domain_id']]
        require(e['model_view_sha256'] == digest(c['model_view']), 'Model view hash mismatch')
        require(c['model_view']['recipient_role'] == c['consumer_view']['recipient_role'] == d['actors']['recipient'], 'Recipient role mismatch')
        require(set(c['consumer_view']['public']) == {'request', 'intake_channel', 'operating_window', 'case_context'}, 'Public schema contains undeclared fields')
        require(set(c['consumer_view']['public_task']) == {'target', 'candidates', 'allowed_checks'}, 'Public task contains undeclared fields')
        require(c['consumer_view']['public_task']['target'] == 'adjudicated_coarse_cause', 'Unexpected task target')
        contexts = c['consumer_view']['downstream_contexts']
        require({v['consumer_id'] for v in contexts} == set(ds.consumers), 'Missing consumer context')
        for ctx in contexts:
            require(set(ctx) == {'consumer_id', 'spec_ref', 'cost_profile_id', 'cost_class_order', 'backend_knows_this_context'}, 'Undeclared downstream context field')
            require(ctx['backend_knows_this_context'] is True, 'Action-only baseline requires backend-visible context')
            require(ctx['cost_class_order'] == class_ids(d), 'Cost column ordering does not match ontology')
            require(ctx['cost_profile_id'] == 'authored_v2', 'Unsupported per-case cost profile; update both oracle and engine explicitly')
            require(ctx['spec_ref'] == 'consumer_specs/' + ctx['consumer_id'] + '.json', 'Incorrect consumer spec reference')
        require(c['model_view']['state']['public'] == c['consumer_view']['public'], 'Public view mismatch')
        require(c['consumer_view']['public_task']['candidates'] == c['model_view']['question']['candidates'], 'Candidate views disagree')
        require(c['consumer_view']['public_task']['allowed_checks'] == d['public_checks'], 'Check specification mismatch')
        require(c['contract_id'] == d['disclosure_contract']['id'], 'Wrong disclosure contract')
        require(set(class_ids(d)) == set((x['id'] for x in c['model_view']['question']['candidates'])), 'Invalid candidates')
        require(e['truth']['S'] in e['hypotheses'] and e['hypotheses'] == sorted(set(e['hypotheses'])), 'Bad secret hypothesis support')
        groups[e['base_id']].append(c)
        lineages[e['lineage_id']].add(e['split'])
        renderings[e['rendering_family_id']].add(e['split'])
    require(all((len(x) == 1 for x in lineages.values())), 'Source lineage crosses splits')
    require(all((len(x) == 1 for x in renderings.values())), 'Declared rendering family crosses splits')
    require(set(groups) == set(certs) == set(reviews) == set((b['base_id'] for b in ds.bases)), 'Missing/extra base, certificate or review')
    checks = 0
    approved = 0
    for b in ds.bases:
        bid = b['base_id']
        worlds = groups[bid]
        require(set(b['case_ids']) == set((c['case_id'] for c in worlds)), 'Base links wrong cases')
        actual = certify_pair(worlds, ds.domains[b['domain_id']], ds.consumers)
        require(actual == certs[bid], 'Stale or fabricated equivalence certificate')
        require(digest(actual) == b['certificate_sha256'], 'Certificate hash mismatch')
        r = reviews[bid]
        errors = list(Draft202012Validator(REVIEW_SCHEMA).iter_errors(r))
        require(not errors, 'Invalid review schema')
        require(r['pair_fingerprint'] == actual['pair_fingerprint'], 'Review applies to different evidence')
        if r['status'] == 'approved':
            require(all((v is True for v in r['checks'].values())), 'Approved review contains incomplete/negative checks')
            require(bool(r['reviewer']) and bool(r['reviewed_at']), 'Approved review lacks reviewer/date')
            if r['review_method'] == 'independent_human':
                approved += 1
        if any((w['evaluator_only']['provenance']['origin'] == 'ai_authored_research_candidate' for w in worlds)):
            require(set(r['checks']) == set(RESEARCH_REVIEW_CHECKS), 'Research candidate review needs every semantic acceptance check')
        if require_reviewed and review_policy == 'independent_human':
            require(r['status'] == 'approved' and r['review_method'] == 'independent_human', 'Independent human approval is required; use --allow-unreviewed only for development')
        checks += len(actual['checks'])
    llm_proof = None
    if review_policy == 'llm_consensus':
        require(judge_consensus is not None, 'LLM review requires a fingerprint-bound consensus artifact')
        require(all((r['status'] == 'approved' and r['review_method'] == 'llm_assisted' for r in ds.reviews)), 'Complete LLM consensus approval required')
        raise ValueError('Semantic judge execution is outside this evaluation release')
    return {'status': 'passed', 'base_pairs': len(groups), 'worlds': len(cases), 'known_source_lineages': len(lineages), 'known_rendering_families': len(renderings), 'pair_invariant_checks': checks, 'independent_human_approved_pairs': approved, 'semantic_status': 'llm_consensus_recorded' if llm_proof else 'independent_review_recorded' if approved == len(groups) else 'unreviewed_or_incomplete', 'review_policy': review_policy, 'llm_consensus_proof': llm_proof, 'llm_approved_pairs': len(groups) if llm_proof else 0, 'scope': 'Schemas + executable contract + frozen hashes + declared lineage checks; no automated proof of natural-language equivalence', 'counts_by_split': dict(Counter((c['evaluator_only']['split'] for c in cases))), 'prototype_reuse_warning': 'Default fixtures reuse v1 mechanism facts; passing does not certify unseen-mechanism or real-workflow generalization' if any((c['evaluator_only']['provenance']['prototype_reuse'] != 'source_family_specific_facts' for c in cases)) else 'Source-specific facts supplied; event distinctness needs separate semantic review; declared lineages do not prove independence'}
