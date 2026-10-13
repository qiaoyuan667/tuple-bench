"""Versioned JSON forest inference without modifying the frozen baseline engine."""
import numpy as np
from .learning import fit_prior, fit_linear, nll, predict as baseline_predict
from .tree_attack import predict_forest
from .io import require

def predict(model, features, classes):
    if model['kind'] != 'random_forest':
        return baseline_predict(model, features, classes)
    probabilities = predict_forest(model, features)
    out = np.zeros((len(features), len(classes)))
    for i, c in enumerate(classes):
        if c in model['classes']:
            out[:, i] = probabilities[:, model['classes'].index(c)]
    require(np.all(np.isfinite(out)) and np.allclose(out.sum(axis=1), 1), 'Bad reconstructed forest posterior')
    return out
