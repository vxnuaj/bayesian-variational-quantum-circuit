"""Classical baselines, trained with scikit-learn."""
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def _n_params_mlp(clf):
    return int(sum(w.size for w in clf.coefs_) + sum(b.size for b in clf.intercepts_))


def classical_baselines(X_tr, y_tr, X_te, y_te, seed=0):
    """Return a list of result dicts for a linear model and two MLPs.

    The small MLP (2 -> 4 -> 1) has 17 parameters, close to the 18-parameter
    circuit, which makes it the fair comparison. The larger one shows the
    ceiling for an unconstrained classical model.
    """
    models = [
        ("Logistic regression (linear)", LogisticRegression()),
        ("MLP 2-4-1 (parameter-matched)", MLPClassifier(hidden_layer_sizes=(4,), activation="tanh", solver="lbfgs",
                                                        max_iter=5000, random_state=seed)),
        ("MLP 2-32-32-1 (large)", MLPClassifier(hidden_layer_sizes=(32, 32), activation="tanh", solver="lbfgs",
                                                max_iter=5000, random_state=seed)),
    ]
    out = []
    for name, clf in models:
        t0 = time.time()
        clf.fit(X_tr, y_tr)
        n = _n_params_mlp(clf) if hasattr(clf, "coefs_") else int(clf.coef_.size + 1)
        out.append(dict(name=name, model=clf, n_params=n,
                        train_acc=float(clf.score(X_tr, y_tr)),
                        test_acc=float(clf.score(X_te, y_te)),
                        seconds=time.time() - t0))
    return out
