"""Train-only, deterministic random forests with JSON-only inference artifacts."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from .io import require
DEFAULT_CANDIDATES = [{'n_estimators': 64, 'max_depth': 3, 'min_samples_leaf': 4}, {'n_estimators': 64, 'max_depth': 6, 'min_samples_leaf': 2}]

def fit_forest(features, labels, classes, parameters, seed):
    require(len(features) == len(labels) > 0 and set(labels) == set(classes), 'Invalid forest training labels')
    require(set(parameters) == {'n_estimators', 'max_depth', 'min_samples_leaf'}, 'Undeclared forest hyperparameter')
    require(all((type(v) is int and v > 0 for v in parameters.values())), 'Positive integer forest parameters required')
    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform(features)
    clf = RandomForestClassifier(**parameters, random_state=seed, n_jobs=1, max_features=1.0)
    clf.fit(X, labels)
    trees = []
    for estimator in clf.estimators_:
        t = estimator.tree_
        values = t.value[:, 0, :]
        prob = values / values.sum(axis=1, keepdims=True)
        trees.append({'left': t.children_left.tolist(), 'right': t.children_right.tolist(), 'feature': t.feature.tolist(), 'threshold': t.threshold.tolist(), 'probabilities': prob.tolist()})
    return {'kind': 'random_forest', 'classes': [v.item() if hasattr(v, 'item') else v for v in clf.classes_], 'feature_names': vec.get_feature_names_out().tolist(), 'trees': trees, 'parameters': parameters, 'seed': seed, 'n_training_rows': len(labels), 'fit_scope': 'train_only', 'serialization': 'JSON_no_pickle'}

def predict_forest(model, features):
    names = model['feature_names']
    index = {k: i for i, k in enumerate(names)}
    X = np.zeros((len(features), len(names)), dtype=np.float32)
    for i, row in enumerate(features):
        for k, v in row.items():
            if k in index:
                X[i, index[k]] = v
    out = np.zeros((len(features), len(model['classes'])))
    require(bool(model['trees']), 'Empty forest')
    for t in model['trees']:
        for i, x in enumerate(X):
            node = 0
            while t['left'][node] != -1:
                node = t['left'][node] if x[t['feature'][node]] <= t['threshold'][node] else t['right'][node]
            out[i] += t['probabilities'][node]
    return out / len(model['trees'])
