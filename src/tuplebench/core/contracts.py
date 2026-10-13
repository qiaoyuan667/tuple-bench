"""Executable business contracts. Cost/check oracles never receive S or a case."""
from __future__ import annotations
import copy
import itertools
from typing import Any
from .io import require, digest, finite_number
CONSUMERS = ('selective_routing', 'one_step_clarification', 'asymmetric_cost_routing')

def class_ids(domain: dict) -> list[str]:
    return [c['id'] for c in domain['classes']]

def validate_specs(domains: dict, consumers: dict, releases: list[dict]) -> None:
    require(set(consumers) == set(CONSUMERS), 'Exactly the three supported consumer types are required')
    for name, c in consumers.items():
        require(c['consumer_id'] == name, 'consumer_id mismatch')
        require(c['oracle_dependencies'] == ['action', 'T', 'public_cost_context'], 'Cost oracle dependency list must exclude secret/private evidence')
    s = consumers['selective_routing']
    q = consumers['one_step_clarification']
    require(s['review_model'] == 'idealized_trusted_correct_T', 'Only declared ideal review is implemented')
    require(q['max_checks'] == 1, 'This engine implements exactly one optional check')
    for x in [s['wrong_cost'], s['review_cost'], q['wrong_cost'], q['check_cost']]:
        require(finite_number(x) and x >= 0, 'Invalid nonnegative business cost')
    for name, d in domains.items():
        require(d['domain_id'] == name, 'domain_id mismatch')
        ids = class_ids(d)
        require(len(ids) == len(set(ids)) and len(ids) >= 2, 'Invalid class list')
        require(set(d['secret_support_by_T']) == set(ids), 'Secret support must cover every T')
        for t, secrets in d['secret_support_by_T'].items():
            require(len(secrets) >= 2 and len(secrets) == len(set(secrets)), 'Need >=2 distinct secrets per T')
            for secret in secrets:
                require(d['business_oracle']['leaf_to_parent'].get(secret) == t, 'Invalid S-to-T ontology')
                require(secret in d['narrative_prototypes'], 'Missing evidence prototype')
        require(d['disclosure_contract']['privacy_condition_on'] == ['X', 'B', 'T'], 'Unexpected privacy contract')
        require(d['disclosure_contract']['protected_variable'] == 'S', 'The implemented target is the declared fine secret S')
        require(d['disclosure_contract']['preapproved_disclosures'] == ['T', 'approved_check_answers_with_declared_law'], 'Do not silently declassify score-dependent actions or S')
        require(d['disclosure_contract']['native_task_authorized'] is True, 'This track authorizes the coarse task')
        require(d['disclosure_contract']['T_reveal_time'] == 'after_business_decision', 'T must not be revealed before utility decision')
        require(d['check_contract']['depends_on'] == ['check_id', 'T'], 'Check oracle may not read S')
        checks = d['public_checks']
        require(len({a['check_id'] for a in checks}) == len(checks), 'Duplicate check')
        for check in checks:
            require(set(check['p_yes_given_T']) == set(ids), 'Incomplete answer likelihood')
            require(all((finite_number(v) and 0 <= v <= 1 for v in check['p_yes_given_T'].values())), 'Invalid answer probability')
        matrix = consumers['asymmetric_cost_routing']['cost_matrix_by_domain'][name]
        require(len(matrix) == len(ids) + 1 and all((len(row) == len(ids) for row in matrix)), 'Cost matrix dimensions')
        require(all((finite_number(v) and v >= 0 for row in matrix for v in row)), 'Invalid matrix cost')
    require(len({r['id'] for r in releases}) == len(releases), 'Duplicate release id')
    expected = {'public_only': [], 'choice': ['choice'], 'choice_confidence': ['choice', 'confidence'], 'choice_max_probability': ['choice', 'max_probability'], 'choice_margin': ['choice', 'margin'], 'choice_top2_ids': ['choice', 'top2_ids'], 'choice_top2_scores': ['choice', 'top2'], 'full_probabilities': ['choice', 'probabilities'], 'full_native': ['choice', 'probabilities', 'confidence']}
    require({r['id'] for r in releases} == set(expected) | {'backend_action_only', 'oracle_T_only'}, 'Release catalog must match implemented mechanisms; extend code and tests for new mechanisms')
    for r in releases:
        if r['id'] in expected:
            require(r['kind'] == 'projection' and r['fields'] == expected[r['id']], 'Release spec disagrees with actual field projection')
        if r['id'] == 'choice_top2_scores':
            require(r['renormalize'] is False and r['omitted'] == 'absent_not_zero', 'Unsupported top-2 representation')

def check_law(domain: dict, check_id: str, T: str) -> dict[str, float]:
    require(T in class_ids(domain), 'Unknown coarse class')
    checks = {r['check_id']: r for r in domain['public_checks']}
    require(check_id in checks, 'Unapproved check')
    p = checks[check_id]['p_yes_given_T'][T]
    return {'yes': float(p), 'no': float(1 - p)}

def cost_vector(consumer: dict, domain: dict, T: str) -> dict[str, float]:
    """Immediate costs; clarification includes every contingent one-check plan."""
    labels = class_ids(domain)
    require(T in labels, 'Unknown T in utility oracle')
    kind = consumer['consumer_id']
    if kind == 'asymmetric_cost_routing':
        mat = consumer['cost_matrix_by_domain'][domain['domain_id']]
        idx = labels.index(T)
        return {**{f'ROUTE:{a}': float(mat[i][idx]) for i, a in enumerate(labels)}, 'GENERAL_INSPECTION': float(mat[-1][idx])}
    wrong = float(consumer['wrong_cost'])
    out = {f'ROUTE:{a}': 0.0 if a == T else wrong for a in labels}
    if kind == 'selective_routing':
        out['REVIEW'] = float(consumer['review_cost'])
    elif kind == 'one_step_clarification':
        for chk in domain['public_checks']:
            law = check_law(domain, chk['check_id'], T)
            for a_yes, a_no in itertools.product(labels, repeat=2):
                key = f"CHECK:{chk['check_id']}|yes:{a_yes}|no:{a_no}"
                out[key] = float(consumer['check_cost'] + law['yes'] * (0 if a_yes == T else wrong) + law['no'] * (0 if a_no == T else wrong))
    else:
        raise ValueError(f'Unknown consumer {kind}')
    return out

def pair_fingerprint(worlds: list[dict]) -> str:
    return digest([{'case_id': w['case_id'], 'model_view': w['model_view'], 'consumer_view': w['consumer_view'], 'truth': w['evaluator_only']['truth']} for w in sorted(worlds, key=lambda v: v['case_id'])])

def certify_pair(worlds: list[dict], domain: dict, consumers: dict) -> dict:
    """Recomputed certificate: declared simulation equivalence, NOT a semantic proof."""
    require(len(worlds) == 2, 'A v2 pair contains exactly two worlds')
    a, b = worlds
    ea, eb = (a['evaluator_only'], b['evaluator_only'])
    require(ea['truth']['S'] != eb['truth']['S'], 'Secrets must differ')
    require(ea['truth']['T'] == eb['truth']['T'], 'Coarse outcome differs')
    require(a['consumer_view'] == b['consumer_view'], 'Public consumer observations differ')
    require(a['model_view']['question'] == b['model_view']['question'], 'Candidate definitions/order differ')
    ma, mb = (copy.deepcopy(a['model_view']), copy.deepcopy(b['model_view']))
    ra = ma['state'].pop('internal_report')
    rb = mb['state'].pop('internal_report')
    require(ma == mb, 'Non-intervened model fields differ')
    require(ra != rb, 'Private evidence did not change')
    for field in ['base_id', 'lineage_id', 'split', 'hypotheses', 'approved_check_answers']:
        require(ea[field] == eb[field], f'Pair mismatch: {field}')
    details = []
    for w in worlds:
        e = w['evaluator_only']
        T, S = (e['truth']['T'], e['truth']['S'])
        require(domain['business_oracle']['leaf_to_parent'].get(S) == T, 'World contradicts ontology')
        law = {q['check_id']: check_law(domain, q['check_id'], T) for q in domain['public_checks']}
        for qid, value in e['approved_check_answers'].items():
            require(qid in law and value in law[qid] and (law[qid][value] > 0), 'Invalid sampled approved answer')
        require(set(e['approved_check_answers']) == set(law), 'Missing approved answer')
        costs = {name: cost_vector(c, domain, T) for name, c in consumers.items()}
        details.append({'T': T, 'all_plan_cost_vectors': costs, 'approved_check_laws': law, 'sampled_check_answers': e['approved_check_answers']})
    require(details[0] == details[1], 'Different business cost or check distributions')
    return {'schema_version': '2.0', 'base_id': ea['base_id'], 'pair_fingerprint': pair_fingerprint(worlds), 'spec_sha256': digest({'domain': domain, 'consumers': consumers}), 'structural_status': 'passed', 'semantic_status': 'requires_separate_human_review', 'checks': {'same_X_B': True, 'same_T': True, 'same_all_action_plan_costs': True, 'same_check_laws': True, 'same_coupled_check_answers': True, 'same_candidates_and_order': True, 'different_S': True}, 'world0_business_signature': details[0], 'world1_business_signature': details[1], 'scope': 'Executable declared initial-dispatch contract only; not all real-world consequences or equal evidence informativeness'}
