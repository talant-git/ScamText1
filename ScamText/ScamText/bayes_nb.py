"""AI Fundamentals, 6-апта: Probability / Bayesian inference.

Naive Bayes classifier written from scratch (no sklearn). It shows Bayes' rule step by step:
    posterior odds = prior odds x  Π_w  P(w | Spam) / P(w | Ham)
The user can change the prior P(Spam) and see how the posterior changes.
"""
import math
import re
from collections import Counter

import numpy as np

# CJK runs become character bigrams (no spaces in Chinese); other scripts become words.
_TOKEN = re.compile(r"[\u3400-\u9fff]+|[^\W\u3400-\u9fff_]+")


def tokenize(clean_text: str):
    """`clean_text` is the output of text_utils.preprocess (same preprocessing as the other models)."""
    tokens = []
    for m in _TOKEN.finditer(clean_text):
        s = m.group()
        if "\u3400" <= s[0] <= "\u9fff":
            tokens += [s[i:i + 2] for i in range(len(s) - 1)] if len(s) > 1 else [s]
        else:
            tokens.append(s)
    return tokens


class NaiveBayes:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha          # Laplace smoothing
        self.vocab = {}             # token -> column index
        self.log_lik = None         # shape (2, V): log P(token | class), class 0=Ham, 1=Spam
        self.prior = 0.5            # P(Spam) estimated from the training data

    # ---- training: just counting -------------------------------------------------
    def fit(self, texts, y):
        counts = (Counter(), Counter())
        docs = [0, 0]
        for t, label in zip(texts, y):
            counts[int(label)].update(tokenize(t))
            docs[int(label)] += 1
        words = sorted(set(counts[0]) | set(counts[1]))
        self.vocab = {w: i for i, w in enumerate(words)}
        n = np.array([[counts[c].get(w, 0) for w in words] for c in (0, 1)], dtype=float)
        # P(w|c) = (count(w,c) + alpha) / (total tokens in c + alpha*V)
        self.log_lik = np.log((n + self.alpha) / (n.sum(axis=1, keepdims=True) + self.alpha * len(words)))
        self.prior = docs[1] / max(1, sum(docs))
        return self

    # ---- inference ---------------------------------------------------------------
    def explain(self, clean_text: str, prior=None, top: int = 6):
        """Bayes' rule on one message. Returns every intermediate number so the UI can show the steps."""
        p = self.prior if prior is None else float(prior)
        if not 0.0 < p < 1.0:
            raise ValueError("prior must be strictly between 0 and 1")
        toks = tokenize(clean_text)
        known = Counter(t for t in toks if t in self.vocab)
        evidence = []
        total = 0.0
        for w, c in known.items():
            i = self.vocab[w]
            llr = c * float(self.log_lik[1, i] - self.log_lik[0, i])   # > 0 pushes towards Spam
            total += llr
            evidence.append({"token": w, "count": c, "llr": round(llr, 3)})
        evidence.sort(key=lambda e: -abs(e["llr"]))
        prior_logodds = math.log(p / (1 - p))
        post_logodds = prior_logodds + total
        posterior = 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, post_logodds))))
        return {
            "prior": round(p, 4),
            "prior_odds": round(p / (1 - p), 4),
            "log10_bayes_factor": round(total / math.log(10), 3),   # evidence from the words (log10 of the likelihood ratio)
            "posterior": round(posterior, 4),
            "tokens_total": len(toks),
            "tokens_known": int(sum(known.values())),
            "tokens_unknown": len(toks) - int(sum(known.values())),
            "evidence": evidence[:top],
        }

    def predict_proba(self, clean_texts, prior=None):
        return np.array([self.explain(t, prior, top=0)["posterior"] for t in clean_texts])

    # ---- persistence (JSON, no pickle) ---------------------------------------------
    def to_dict(self):
        return {"alpha": self.alpha, "prior": self.prior, "vocab": list(self.vocab),
                "log_lik": np.round(self.log_lik, 6).tolist()}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["alpha"])
        m.prior = d["prior"]
        m.vocab = {w: i for i, w in enumerate(d["vocab"])}
        m.log_lik = np.array(d["log_lik"], dtype=float)
        return m
