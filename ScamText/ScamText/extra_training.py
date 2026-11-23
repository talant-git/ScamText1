"""QAIRU AI Fundamentals Final: trains the additional from-scratch models on the SAME split as train_model.py.

Called at the end of train_model.main(). It only ADDS keys ("bayes", "neural_net", "safety") to the report and
writes new files; the Logistic Regression model, its metrics and its files are not touched.
  6-апта  : bayes_nb.NaiveBayes   (Bayes' rule, written from scratch)
  9-апта  : neural_net.MLP        (backpropagation, written from scratch)
  10-апта : safety.evaluate       (evasion attacks on all three models)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from text_utils import preprocess
from bayes_nb import NaiveBayes
from neural_net import MLP
import safety

SEED = 42
HIDDEN_GRID = (8, 16, 32)       # chosen on a validation split taken from TRAIN only (never on the test set)
MAX_EPOCHS, LR, L2, INIT_STD = 300, 0.02, 1e-4, 0.1


def _metrics(y_true, y_pred):
    return {"accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4), "support": int(len(y_true))}


def _evaluate(test, proba):
    pred = (proba >= 0.5).astype(int)
    out = {"overall": _metrics(test.y, pred), "per_language": {}}
    for lang in sorted(test.Language.unique()):
        mask = (test.Language == lang).values
        out["per_language"][lang] = _metrics(test.y.values[mask], pred[mask])
    out["confusion_matrix"] = {"labels": ["Ham", "Spam"], "matrix": confusion_matrix(test.y, pred, labels=[0, 1]).tolist()}
    return out


def _balanced_weights(y):
    y = np.asarray(y); n = len(y); n1 = max(1, int(y.sum())); n0 = max(1, n - n1)
    return np.where(y == 1, n / (2 * n1), n / (2 * n0))


def train_extra(base, train, test, train_aug, vec, clf, report, augment, make_vectorizer):
    models_dir, results_dir = base / "models", base / "results"

    # ---------------- 6-апта: Naive Bayes / Bayes' rule ----------------
    nb = NaiveBayes(alpha=1.0).fit(train_aug.clean.map(preprocess), train_aug.y)
    test_clean = test.clean.map(preprocess)
    p_nb = nb.predict_proba(test_clean)
    report["bayes"] = {
        "week": 6, "algorithm": "Naive Bayes (Bayes ережесі, нөлден жазылған), Laplace α=1",
        "vocab_size": len(nb.vocab), "train_prior_spam": round(nb.prior, 4), **_evaluate(test, p_nb)}
    (models_dir / "nb_model.json").write_text(json.dumps(nb.to_dict()), encoding="utf-8")

    # ---------------- 9-апта: neural network ----------------
    # model selection on a validation split of TRAIN (split BEFORE augmentation, so no text leaks between parts)
    tr_sub, val = train_test_split(train, test_size=0.15, random_state=SEED, stratify=train.strat)
    v_s = make_vectorizer()
    aug = augment(tr_sub)
    Xs = v_s.fit_transform(aug.clean.map(preprocess)); Xv = v_s.transform(val.clean.map(preprocess))
    selection = []
    for h in HIDDEN_GRID:
        net = MLP(Xs.shape[1], hidden=h, lr=LR, l2=L2, init_std=INIT_STD, seed=SEED)
        net.fit(Xs, aug.y.values, epochs=MAX_EPOCHS, sample_weight=_balanced_weights(aug.y), X_val=Xv, y_val=val.y.values)
        best = int(np.argmin(net.val_loss_history))
        selection.append({"hidden": h, "best_epoch": best + 1, "val_loss": round(net.val_loss_history[best], 4)})
    best_cfg = min(selection, key=lambda s: s["val_loss"])

    X_tr = vec.transform(train_aug.clean.map(preprocess)); X_te = vec.transform(test_clean)
    net = MLP(X_tr.shape[1], hidden=best_cfg["hidden"], lr=LR, l2=L2, init_std=INIT_STD, seed=SEED)
    net.fit(X_tr, train_aug.y.values, epochs=best_cfg["best_epoch"], sample_weight=_balanced_weights(train_aug.y))
    p_nn = net.predict_proba(X_te)
    report["neural_net"] = {
        "week": 9, "algorithm": "Көпқабатты нейрондық желі (backpropagation + Adam, нөлден жазылған)",
        "architecture": f"{X_tr.shape[1]} TF-IDF кіріс → {net.hidden} жасырын нейрон (tanh) → 1 шығыс (sigmoid)",
        "n_parameters": int(net.W1.size + net.b1.size + net.W2.size + 1),
        "learning_rate": LR, "l2": L2, "epochs": best_cfg["best_epoch"], "hidden": net.hidden,
        "model_selection_on_validation": selection, **_evaluate(test, p_nn)}
    report["neural_net"]["loss_curve"] = [round(x, 4) for x in net.loss_history[::5]]   # every 5th epoch, for the web chart
    net.save(models_dir / "nn_weights.npz")

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(range(1, len(net.loss_history) + 1), net.loss_history, color="#8b5cf6")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss (BCE)"); ax.set_title("Neural network training loss")
    fig.tight_layout(); fig.savefig(results_dir / "nn_loss.png", dpi=150); plt.close(fig)

    # ---------------- 10-апта: AI safety (evasion attacks) ----------------
    lr_fn = lambda ts: clf.predict_proba(vec.transform([preprocess(t) for t in ts]))[:, 1]
    nb_fn = lambda ts: nb.predict_proba([preprocess(t) for t in ts])
    nn_fn = lambda ts: net.predict_proba(vec.transform([preprocess(t) for t in ts]))
    spam_texts = test[test.y == 1].clean.tolist()
    report["safety"] = {"description": "Spam recall on test Spam texts that the attack changed: before vs after the attack.",
                        "n_spam_test": len(spam_texts),
                        "attacks": safety.evaluate({"Logistic Regression": lr_fn, "Naive Bayes": nb_fn, "Neural Network": nn_fn}, spam_texts)}
