"""Four independent score ablations; inference receives one response only."""
import copy
import numpy as np
from tuplebench.core.io import require
from tuplebench.core.releases import validate_response
METHODS = ['full_native', 'quantized_10', 'top2', 'temperature_2', 'noise_05_seed1', 'noise_05_seed2', 'noise_05_seed3']
NOISE_SEEDS = dict(noise_05_seed1=20261011, noise_05_seed2=20261012, noise_05_seed3=20261013)
DISPLAY = dict(full_native='Full native', quantized_10='0.1 probability quantization', top2='Top-2 scores', temperature_2='Temperature (tau=2; inverse-aware attack)', noise_05_seed1='Gaussian noise: seed 1', noise_05_seed2='Gaussian noise: seed 2', noise_05_seed3='Gaussian noise: seed 3')

def transform(response, candidate_order, method, rng=None):
    require(method in METHODS, 'Unknown ablation')
    require(set(response) == {'choice', 'probabilities', 'confidence'}, 'Undeclared input')
    validate_response(response, candidate_order)
    out = copy.deepcopy(response)
    if method == 'full_native':
        return out
    p = np.array([response['probabilities'][k] for k in candidate_order], dtype=float)
    if method == 'top2':
        ranks = sorted(range(len(p)), key=lambda i: (-p[i], i))[:2]
        return dict(choice=response['choice'], confidence=response['confidence'], top2=[dict(id=candidate_order[i], score=float(p[i])) for i in ranks])
    if method == 'quantized_10':
        raw = p * 10
        counts = np.floor(raw).astype(int)
        remaining = 10 - int(counts.sum())
        require(0 <= remaining <= len(p), 'Invalid simplex remainder')
        ranks = sorted(range(len(p)), key=lambda i: (-(raw[i] - counts[i]), i))
        counts[ranks[:remaining]] += 1
        p = counts / 10
    elif method == 'temperature_2':
        p = np.sqrt(p)
        p /= p.sum()
    else:
        require(rng is not None, 'Noise needs an independent random generator')
        p = np.maximum(p + rng.normal(0, 0.05, len(p)), 0)
        p = p / p.sum() if p.sum() > 0 else np.ones(len(p)) / len(p)
    out['probabilities'] = dict(zip(candidate_order, p.tolist()))
    validate_response(out, candidate_order)
    return out

def inverse_temperature(payload):
    """Known inverse, using released fields alone; zero scores stay zero."""
    out = copy.deepcopy(payload)
    keys = list(payload['probabilities'])
    p = np.array([payload['probabilities'][k] for k in keys]) ** 2
    require(p.sum() > 0, 'Empty temperature scores')
    out['probabilities'] = dict(zip(keys, (p / p.sum()).tolist()))
    return out
