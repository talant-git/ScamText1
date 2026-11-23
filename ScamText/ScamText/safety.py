"""AI Fundamentals, 10-апта: AI Safety — robustness against evasion (adversarial text).

Spammers change spelling to dodge filters. Each function keeps the message readable for a human.
We measure how much each model's recall on Spam drops (see extra_training.py).
"""
_HOMOGLYPH = {"а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y", "і": "i", "к": "k"}
_LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5"}


def leetspeak(text):
    return "".join(_LEET.get(c.lower(), c) for c in text)


def homoglyph(text):
    """Cyrillic letters -> Latin lookalikes (looks the same to a human, different characters to a model)."""
    out = []
    for c in text:
        r = _HOMOGLYPH.get(c.lower())
        out.append((r.upper() if c.isupper() else r) if r else c)
    return "".join(out)


def spacing(text):
    """Insert dots inside long words ("prize" -> "p.r.i.z.e"); for Chinese put spaces between characters."""
    out = []
    for w in text.split(" "):
        if any("\u3400" <= c <= "\u9fff" for c in w):
            out.append(" ".join(w))
        elif len(w) >= 5:
            out.append(".".join(w))
        else:
            out.append(w)
    return " ".join(out)


PERTURBATIONS = {
    "leetspeak": (leetspeak, "Leetspeak (a→4, e→3, o→0 ...)"),
    "homoglyph": (homoglyph, "Homoglyph (кирилл әріптері → латын ұқсастары)"),
    "spacing": (spacing, "Нүкте/бос орын қосу (p.r.i.z.e)"),
}


def evaluate(models, spam_texts):
    """models: {name: f(list[str]) -> np.array of P(Spam)}. Recall on the texts that the attack actually changed."""
    result = {}
    for key, (fn, label) in PERTURBATIONS.items():
        changed = [t for t in spam_texts if fn(t) != t]
        perturbed = [fn(t) for t in changed]
        row = {"label": label, "n_changed": len(changed), "models": {}}
        for name, proba in models.items():
            if changed:
                row["models"][name] = {"recall_clean": round(float((proba(changed) >= 0.5).mean()), 4),
                                       "recall_attacked": round(float((proba(perturbed) >= 0.5).mean()), 4)}
        result[key] = row
    return result
