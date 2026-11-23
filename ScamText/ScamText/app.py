"""ScamText Flask server. Runs locally; user messages are never stored or logged."""
import json, pickle, re
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import numpy as np
from text_utils import preprocess, detect_language
from bayes_nb import NaiveBayes          # 6-апта
from neural_net import MLP, xor_demo     # 9-апта
import safety                            # 10-апта

BASE = Path(__file__).parent
MAX_CHARS = 1000
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024

with open(BASE / "models" / "vectorizer.pkl", "rb") as f:
    VEC = pickle.load(f)
with open(BASE / "models" / "scamtext_model.pkl", "rb") as f:
    CLF = pickle.load(f)
METRICS = json.loads((BASE / "results" / "metrics.json").read_text(encoding="utf-8"))
LANGS = {"auto", "Kazakh", "Russian", "English", "Chinese"}

try:   # models written by train_model.py (extra_training.py); the original analyzer works without them
    NB = NaiveBayes.from_dict(json.loads((BASE / "models" / "nb_model.json").read_text(encoding="utf-8")))
    NN = MLP.load(BASE / "models" / "nn_weights.npz")
except (OSError, ValueError, KeyError):
    NB = NN = None


def spam_proba(texts):
    return CLF.predict_proba(VEC.transform([preprocess(t) for t in texts]))[:, 1]


def explain(text, is_spam, top=3):
    """Leave-one-word-out: which words moved the model most toward its verdict."""
    chinese = detect_language(text) == "Chinese"
    words = [w for w in re.split(r"[，。！？；：、（）【】「」“”,.!?;:()\[\]\s]+", text) if w] if chinese else text.split()
    sep = "，" if chinese else " "
    if len(words) < 2:
        return []
    base = spam_proba([text])[0]
    variants = [sep.join(words[:i] + words[i + 1:]) for i in range(len(words))]
    shifts = base - spam_proba(variants)          # >0: word pushes toward Spam
    shifts = shifts if is_spam else -shifts       # orient toward predicted class
    order = np.argsort(-shifts)[:top]
    return [words[i].strip(".,!?:;()[]\"'") for i in order if shifts[i] > 0.005]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyzer")
def analyzer():
    return render_template("analyzer.html", max_chars=MAX_CHARS)


@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")


@app.route("/performance")
def performance():
    return render_template("performance.html", m=METRICS)


@app.route("/about")
def about():
    return render_template("about.html", m=METRICS)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    lang = data.get("language", "auto")
    if not isinstance(text, str) or not text.strip():
        return jsonify(error="Мәтін бос. Тексеру үшін хабарлама енгізіңіз."), 400
    text = text.strip()
    if len(text) > MAX_CHARS:
        return jsonify(error=f"Мәтін тым ұзын. Ең көбі {MAX_CHARS} таңба."), 400
    if lang not in LANGS:
        return jsonify(error="Белгісіз тіл таңдалды."), 400
    try:
        p = float(spam_proba([text])[0])
        is_spam = p >= 0.5
        detected = detect_language(text)
        return jsonify(
            label="Spam" if is_spam else "Ham",
            language=detected if lang == "auto" else lang,
            language_detected=detected,
            confidence=round(p if is_spam else 1 - p, 4),
            spam_probability=round(p, 4),
            suspicious_words=explain(text, is_spam),
        )
    except Exception:
        return jsonify(error="Талдау кезінде қате орын алды. Қайталап көріңіз."), 500


# ---------------------------------------------------------------------------------------------
# QAIRU AI Fundamentals Final: "AI зертханасы" (6-апта Bayes, 8-апта LR, 9-апта NN, 10-апта safety)
# ---------------------------------------------------------------------------------------------
def nn_proba(texts):
    return NN.predict_proba(VEC.transform([preprocess(t) for t in texts]))


def explain_with(proba_fn, text, is_spam, top=3):
    """Same leave-one-word-out idea as explain(), but for any model's probability function."""
    chinese = detect_language(text) == "Chinese"
    words = [w for w in re.split(r"[，。！？；：、（）【】「」“”,.!?;:()\[\]\s]+", text) if w] if chinese else text.split()
    sep = "，" if chinese else " "
    if len(words) < 2:
        return []
    base = proba_fn([text])[0]
    shifts = base - proba_fn([sep.join(words[:i] + words[i + 1:]) for i in range(len(words))])
    shifts = shifts if is_spam else -shifts
    return [words[i].strip(".,!?:;()[]\"'") for i in np.argsort(-shifts)[:top] if shifts[i] > 0.005]


def _text_or_error(data):
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        return None, "Мәтін бос. Тексеру үшін хабарлама енгізіңіз."
    text = text.strip()
    if len(text) > MAX_CHARS:
        return None, f"Мәтін тым ұзын. Ең көбі {MAX_CHARS} таңба."
    return text, None


def _number(v, lo, hi, integer=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or not lo <= v <= hi:
        return None
    if integer and int(v) != v:
        return None
    return int(v) if integer else float(v)


@app.route("/lab")
def lab():
    return render_template("lab.html", m=METRICS, max_chars=MAX_CHARS, ready=NB is not None and "bayes" in METRICS)


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """One message -> three models: Logistic Regression (8), Naive Bayes with adjustable prior (6), neural network (9)."""
    data = request.get_json(silent=True) or {}
    text, err = _text_or_error(data)
    if err:
        return jsonify(error=err), 400
    prior = data.get("prior")
    if prior is not None:
        prior = _number(prior, 0.01, 0.99)
        if prior is None:
            return jsonify(error="Prior P(Spam) 0,01 мен 0,99 аралығында болуы керек."), 400
    if NB is None or NN is None:
        return jsonify(error="Қосымша модельдер табылмады. Алдымен python train_model.py іске қосыңыз."), 503
    try:
        p_lr = float(spam_proba([text])[0])
        bayes = NB.explain(preprocess(text), prior)
        p_nn = float(nn_proba([text])[0])
        return jsonify(
            language=detect_language(text),
            logistic_regression={"label": "Spam" if p_lr >= 0.5 else "Ham", "spam_probability": round(p_lr, 4),
                                 "suspicious_words": explain(text, p_lr >= 0.5)},
            naive_bayes={"label": "Spam" if bayes["posterior"] >= 0.5 else "Ham", **bayes},
            neural_net={"label": "Spam" if p_nn >= 0.5 else "Ham", "spam_probability": round(p_nn, 4),
                        "suspicious_words": explain_with(nn_proba, text, p_nn >= 0.5)},
        )
    except Exception:
        return jsonify(error="Талдау кезінде қате орын алды. Қайталап көріңіз."), 500


@app.route("/api/safety", methods=["POST"])
def api_safety():
    """Apply an evasion attack to the message and show how each model's verdict changes."""
    data = request.get_json(silent=True) or {}
    text, err = _text_or_error(data)
    if err:
        return jsonify(error=err), 400
    attack = data.get("attack")
    if attack not in safety.PERTURBATIONS:
        return jsonify(error="Белгісіз шабуыл түрі."), 400
    if NB is None or NN is None:
        return jsonify(error="Қосымша модельдер табылмады. Алдымен python train_model.py іске қосыңыз."), 503
    fn, label = safety.PERTURBATIONS[attack]
    attacked = fn(text)

    def probs(t):
        return {"Logistic Regression": float(spam_proba([t])[0]), "Naive Bayes": float(NB.predict_proba([preprocess(t)])[0]),
                "Neural Network": float(nn_proba([t])[0])}
    before, after = probs(text), probs(attacked)
    return jsonify(attack=label, changed=attacked != text, original_text=text, attacked_text=attacked,
                   results=[{"model": k, "before": round(before[k], 4), "after": round(after[k], 4),
                             "label_before": "Spam" if before[k] >= 0.5 else "Ham",
                             "label_after": "Spam" if after[k] >= 0.5 else "Ham"} for k in before])


@app.route("/api/xor", methods=["POST"])
def api_xor():
    """Perceptron vs neural network on XOR; the user chooses the network's parameters."""
    data = request.get_json(silent=True) or {}
    hidden = _number(data.get("hidden", 4), 1, 16, integer=True)
    epochs = _number(data.get("epochs", 1000), 10, 5000, integer=True)
    lr = _number(data.get("lr", 0.05), 0.001, 1.0)
    seed = _number(data.get("seed", 1), 0, 1000, integer=True)
    if None in (hidden, epochs, lr, seed):
        return jsonify(error="Параметр дұрыс емес: жасырын нейрон 1–16, epoch 10–5000, learning rate 0,001–1, seed 0–1000."), 400
    return jsonify(xor_demo(hidden=hidden, lr=lr, epochs=epochs, seed=seed))


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Сұраныс тым үлкен."), 413


@app.errorhandler(404)
def not_found(_):
    return render_template("index.html"), 404


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
