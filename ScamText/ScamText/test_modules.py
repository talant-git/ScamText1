"""Tests for the Final Project modules: 6-апта Naive Bayes, 9-апта neural network, 10-апта safety, and their API.
Run: python test_modules.py   (result is written to results/test_modules.txt)"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import safety
from bayes_nb import NaiveBayes, tokenize
from neural_net import MLP, Perceptron, XOR_X, XOR_Y, xor_demo
from text_utils import preprocess

BASE = Path(__file__).parent
out, failed = [], 0


def check(name, ok, detail=""):
    global failed
    failed += not ok
    line = f"{'OK  ' if ok else 'ҚАТЕ'} | {name}" + (f" | {detail}" if detail else "")
    out.append(line); print(line)


def section(t):
    out.append(""); out.append(f"== {t}"); print(); print(f"== {t}")


# ---------------------------------------------------------------- 6-апта: Naive Bayes
section("6-апта: Naive Bayes (нөлден жазылған) sklearn нәтижесімен салыстыру")
df = pd.read_csv(BASE / "data" / "kazakh_gemini_300.csv").dropna(subset=["Message"])
texts = df.Message.map(preprocess).tolist()
y = (df.Label == "Spam").astype(int).to_numpy()
mine = NaiveBayes(alpha=1.0).fit(texts, y)
cv = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
ref = MultinomialNB(alpha=1.0).fit(cv.fit_transform(texts), y)
p_ref = ref.predict_proba(cv.transform(texts))[:, 1]
p_my = mine.predict_proba(texts)
check("Posterior sklearn MultinomialNB-мен сәйкес", np.allclose(p_my, p_ref, atol=6e-5),
      f"ең үлкен айырма = {np.abs(p_my - p_ref).max():.2e}, {len(texts)} мәтін")
check("Оқыту деректеріндегі жапсырмалар бірдей болжанады", ((p_my >= 0.5) == (p_ref >= 0.5)).all())

e1, e2, e3 = (mine.explain(texts[0], prior=p) for p in (0.1, 0.5, 0.9))
check("Prior өссе, posterior өседі", e1["posterior"] <= e2["posterior"] <= e3["posterior"],
      f"{e1['posterior']} ≤ {e2['posterior']} ≤ {e3['posterior']}")
expected = 1 / (1 + math.exp(-e2["log10_bayes_factor"] * math.log(10)))
check("prior=0,5 болғанда posterior = sigmoid(дәлел)", abs(e2["posterior"] - expected) < 1e-3)
check("Танылмаған сөздер ескерілмейді", mine.explain("zzzqqq xxyyww")["tokens_known"] == 0
      and abs(mine.explain("zzzqqq xxyyww", prior=0.3)["posterior"] - 0.3) < 1e-3)
for bad in (0, 1, -0.2, 1.5):
    try:
        mine.explain("test", prior=bad); check(f"Қате prior={bad} қабылданбайды", False)
    except ValueError:
        check(f"Қате prior={bad} қабылданбайды", True)
check("Қытай мәтіні биграмға бөлінеді", tokenize("恭喜您中奖") == ["恭喜", "喜您", "您中", "中奖"])

# ---------------------------------------------------------------- 9-апта: neural network
section("9-апта: нейрондық желі (нөлден жазылған)")
rng = np.random.default_rng(0)
Xg, yg = rng.normal(size=(12, 5)), rng.integers(0, 2, 12).astype(float)
net = MLP(5, hidden=3, lr=0.01, l2=1e-2, init_std=0.5, seed=1)
_, (gW1, gb1, gW2, gb2) = net.loss_and_grads(Xg, yg)


def loss_now():
    return net.loss_and_grads(Xg, yg)[0]


def numeric(arr, idx, eps=1e-6):
    old = arr[idx]; arr[idx] = old + eps; lp = loss_now(); arr[idx] = old - eps; lm = loss_now(); arr[idx] = old
    return (lp - lm) / (2 * eps)


worst = 0.0
for arr, g in ((net.W1, gW1), (net.b1, gb1), (net.W2, gW2)):
    for idx in np.ndindex(arr.shape):
        a, n = g[idx], numeric(arr, idx)
        worst = max(worst, abs(a - n) / max(1e-8, abs(a) + abs(n)))
b2_old = net.b2; net.b2 = b2_old + 1e-6; lp = loss_now(); net.b2 = b2_old - 1e-6; lm = loss_now(); net.b2 = b2_old
worst = max(worst, abs(gb2 - (lp - lm) / 2e-6) / max(1e-8, abs(gb2) + abs((lp - lm) / 2e-6)))
check("Backpropagation градиенті сандық градиентпен сәйкес", worst < 1e-5, f"ең үлкен салыстырмалы қате = {worst:.2e}")

before = net.loss_and_grads(Xg, yg)[0]
net.fit(Xg, yg, epochs=200)
check("Оқыту loss-ты азайтады", net.loss_history[-1] < before, f"{before:.4f} → {net.loss_history[-1]:.4f}")

d = xor_demo(hidden=4, lr=0.05, epochs=1000, seed=1)
check("Нейрондық желі XOR-ды шешеді (4 жасырын нейрон)", d["neural_net"]["accuracy"] == 1.0)
check("Perceptron XOR-ды шеше алмайды (сызықпен бөлінбейді)", d["perceptron"]["accuracy"] < 1.0,
      f"accuracy = {d['perceptron']['accuracy']}")
check("Perceptron AND-ты үйренеді (сызықпен бөлінеді)",
      (Perceptron(2, seed=0).fit(XOR_X, np.array([0, 0, 0, 1]), 100).predict(XOR_X) == np.array([0, 0, 0, 1])).all())

# ---------------------------------------------------------------- 10-апта: safety
section("10-апта: алдау шабуылдарының функциялары")
check("Homoglyph кирилл «а» әрпін латын «a» әрпіне ауыстырады",
      safety.homoglyph("алма") == "a\u043bм" + "a" and safety.homoglyph("алма")[0] == "a" and safety.homoglyph("алма") != "алма")
check("Homoglyph ағылшын мәтінін өзгертпейді", safety.homoglyph("free prize") == "free prize")
check("Spacing ұзын сөзге нүкте қосады", safety.spacing("free prize now") == "free p.r.i.z.e now")
check("Leetspeak әріпті цифрға ауыстырады", safety.leetspeak("free") == "fr33")

# ---------------------------------------------------------------- API
section("API: қосымша endpoint-тер және қате енгізу")
from app import app  # noqa: E402
c = app.test_client()
spam = "Поздравляем! Вам доступен займ 300 000 тенге без проверок, оформите заявку по ссылке сейчас."
r = c.post("/api/compare", json={"text": spam, "prior": 0.3})
d = r.get_json()
check("/api/compare үш модельді қайтарады", r.status_code == 200 and {"logistic_regression", "naive_bayes", "neural_net"} <= set(d))
check("Spam мысалын үш модель де Spam деп таниды",
      d["logistic_regression"]["label"] == d["naive_bayes"]["label"] == d["neural_net"]["label"] == "Spam")
lo = c.post("/api/compare", json={"text": "Привет", "prior": 0.05}).get_json()["naive_bayes"]["posterior"]
hi = c.post("/api/compare", json={"text": "Привет", "prior": 0.95}).get_json()["naive_bayes"]["posterior"]
check("API-да prior өзгерсе posterior өзгереді", lo < hi, f"{lo} < {hi}")
for name, body, code in [
    ("бос мәтін", {"text": "  "}, 400), ("мәтін жоқ", {}, 400), ("1001 таңба", {"text": "a" * 1001}, 400),
    ("prior = 0", {"text": "hi", "prior": 0}, 400), ("prior мәтін", {"text": "hi", "prior": "көп"}, 400),
    ("prior = True", {"text": "hi", "prior": True}, 400), ("дұрыс prior", {"text": "hi", "prior": 0.5}, 200)]:
    check(f"/api/compare: {name}", c.post("/api/compare", json=body).status_code == code)
check("/api/compare: JSON емес сұраныс", c.post("/api/compare", data="x").status_code == 400)

for name, body, code in [
    ("әдепкі параметр", {}, 200), ("hidden=0", {"hidden": 0}, 400), ("hidden=1,5", {"hidden": 1.5}, 400),
    ("epochs=10**9", {"epochs": 10**9}, 400), ("lr мәтін", {"lr": "жылдам"}, 400), ("lr=0", {"lr": 0}, 400),
    ("seed теріс", {"seed": -1}, 400), ("hidden=1 (желі қателеседі, бірақ жұмыс істейді)", {"hidden": 1}, 200)]:
    check(f"/api/xor: {name}", c.post("/api/xor", json=body).status_code == code)

for attack in safety.PERTURBATIONS:
    r = c.post("/api/safety", json={"text": spam, "attack": attack})
    check(f"/api/safety: {attack}", r.status_code == 200 and len(r.get_json()["results"]) == 3)
check("/api/safety: белгісіз шабуыл", c.post("/api/safety", json={"text": "hi", "attack": "x"}).status_code == 400)
check("/api/safety: бос мәтін", c.post("/api/safety", json={"text": "", "attack": "spacing"}).status_code == 400)
r = c.post("/api/compare", json={"text": "<script>alert(1)</script> free prize"})
check("<script> мәтіні қауіпсіз өңделеді (JSON ретінде)", r.status_code == 200 and r.is_json)
check("/lab беті ашылады", c.get("/lab").status_code == 200)
check("Бұрынғы /api/analyze өзгермеген", c.post("/api/analyze", json={"text": spam}).get_json()["label"] == "Spam")

section("Қорытынды")
out.append(f"Сәтсіз тест: {failed}"); print(out[-1])
(BASE / "results" / "test_modules.txt").write_text("\n".join(out), encoding="utf-8")
sys.exit(1 if failed else 0)
