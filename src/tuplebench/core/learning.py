"""Train-only preprocessing; validation-only selection; JSON-portable fits."""
from __future__ import annotations
import warnings
import numpy as np
from scipy.special import expit, softmax
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from .io import require

def fit_prior(labels: list, classes: list) -> dict:
    counts = np.array([sum((v == c for v in labels)) for c in classes], dtype=float) + 1.0
    return {'kind': 'prior', 'classes': classes, 'probabilities': (counts / counts.sum()).tolist()}

def fit_linear(features: list[dict], labels: list, classes: list, C: float) -> dict:
    require(len(features) == len(labels) and len(features) > 0, 'Empty/mismatched training data')
    require(set(labels) == set(classes), 'Training split must contain every declared class')
    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform(features)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    clf = LogisticRegression(C=C, solver='lbfgs', max_iter=2000, tol=1e-06)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        clf.fit(X, labels)
    require(not any((issubclass(w.category, ConvergenceWarning) for w in caught)), 'Logistic regression did not converge; do not silently score this fit')
    return {'kind': 'logistic', 'classes': [x.item() if hasattr(x, 'item') else x for x in clf.classes_], 'feature_names': vec.get_feature_names_out().tolist(), 'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(), 'coef': clf.coef_.tolist(), 'intercept': clf.intercept_.tolist(), 'C': float(C), 'n_training_rows': len(labels), 'fit_scope': 'train_only'}

def predict(model: dict, features: list[dict], classes: list) -> np.ndarray:
    if model['kind'] == 'prior':
        p = np.tile(np.array(model['probabilities']), (len(features), 1))
    else:
        names = model['feature_names']
        lookup = {k: i for i, k in enumerate(names)}
        X = np.zeros((len(features), len(names)), dtype=float)
        for i, row in enumerate(features):
            for k, v in row.items():
                if k in lookup:
                    X[i, lookup[k]] = v
        X = (X - np.array(model['mean'])) / np.array(model['scale'])
        z = X @ np.array(model['coef']).T + np.array(model['intercept'])
        if len(model['classes']) == 2 and z.shape[1] == 1:
            p1 = expit(z[:, 0])
            p = np.column_stack((1 - p1, p1))
        else:
            p = softmax(z, axis=1)
    out = np.zeros((len(features), len(classes)))
    for j, c in enumerate(classes):
        if c in model['classes']:
            out[:, j] = p[:, model['classes'].index(c)]
    require(np.all(np.isfinite(out)) and np.allclose(out.sum(axis=1), 1), 'Bad reconstructed posterior')
    return out

def nll(labels: list, probabilities: np.ndarray, classes: list) -> float:
    if not labels:
        return float('inf')
    idx = [classes.index(y) for y in labels]
    p = probabilities[np.arange(len(idx)), idx]
    return float(-np.log(np.clip(p, 1e-12, 1)).mean())

def smooth_posterior(q: np.ndarray, floor: float) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    require(q.ndim == 1 and np.all(np.isfinite(q)) and np.all(q >= 0), 'Invalid posterior')
    return (q + floor) / (q.sum() + floor * len(q))
