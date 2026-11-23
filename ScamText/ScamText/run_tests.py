"""API tests (no browser needed). Writes the real output to results/test_run.txt."""
import sys
from pathlib import Path
from app import app

CASES = [
    ("Қазақша Spam", "Сізге 200 000 теңге несие бекітілді! Ақшаны алу үшін сілтемеге өтіп, карта деректерін енгізіңіз.", "Spam"),
    ("Қазақша Ham", "Сәлем, ертең сағат 10-да кітапханада кездесеміз бе?", "Ham"),
    ("Русский Spam", "Поздравляем! Вам доступен займ 300 000 тенге без проверок, оформите заявку по ссылке сейчас.", "Spam"),
    ("Русский Ham", "Дорогая, я купил билеты на субботу, поезд отправляется в 7:40 с первой платформы.", "Ham"),
    ("English Spam", "Final notice: your streaming subscription failed. Update billing details here to avoid cancellation.", "Spam"),
    ("English Ham", "Running 10 minutes late, order me a coffee please.", "Ham"),
    ("English Ham (банк хабарламасы)", "Your card ending in 1234 was used for $18.40 at Green Market. If this wasn't you, call the number on the back of your card.", "Ham"),
    ("112 қазақша латынмен", "08.11.26 Astana k. noser zhauyn kutilude", "Ham"),
    ("112 орысша латынмен", "08.11.26 g. Astana ozhidaetsya silnyi dozhd", "Ham"),
    ("中文 Spam (奖品)", "恭喜您获得iPhone一台，点击链接填写收货信息并支付10元运费。", "Spam"),
    ("中文 Spam (刷单)", "兼职刷单，每单返现20元，日结，加客服微信 abc12345", "Spam"),
    ("中文 Ham (朋友)", "晚上我们去吃火锅吧，我已经订好位子了。", "Ham"),
    ("中文 Ham (验证码)", "您的验证码是582910，5分钟内有效，请勿泄露给他人。", "Ham"),
    ("中文 Ham (预警)", "【应急管理】请注意：今夜我市有强降雨，请勿前往低洼地带。", "Ham"),
    ("中文 Ham ('中奖'玩笑)", "我中奖了！哈哈，一等奖是一支冰淇淋。", "Ham"),
    ("繁體中文 Spam", "恭喜您中獎，請點擊連結領取獎金。", "Spam"),
    ("繁體中文 Ham", "明天下午三點開會，請準時到。", "Ham"),
]

out = []
def log(s=""):
    out.append(s); print(s)

c = app.test_client()
wrong = 0
log("== Хабарлама тесттері (күтілген / модель нәтижесі)")
for name, text, exp in CASES:
    d = c.post("/api/analyze", json={"text": text}).get_json()
    ok = d["label"] == exp
    wrong += not ok
    log(f"{'OK  ' if ok else 'ҚАТЕ'} | {name} | күтілген={exp} | нәтиже={d['label']} | P(spam)={d['spam_probability']} | тіл={d['language']}")
log(f"Қате жіктелген: {wrong} / {len(CASES)}")

log("\n== Қате енгізулер")
def status(**kw): 
    r = c.post("/api/analyze", **kw); return r.status_code, (r.get_json() or {}).get("error")
log(f"бос мәтін: {status(json={'text': '   '})}")
log(f"мәтін жоқ: {status(json={})}")
log(f"1001 таңба: {status(json={'text': 'a' * 1001})}")
log(f"JSON емес: {status(data='x')}")
log(f"белгісіз тіл: {status(json={'text': 'hi', 'language': 'xx'})}")
r = c.post("/api/analyze", json={"text": "<script>alert(1)</script> free prize"}).get_json()
log(f"<script> мәтіні қауіпсіз өңделді: label={r['label']}, сөздер={r['suspicious_words']}")
r = c.post("/api/analyze", json={"text": "x" * 1000})
log(f"дәл 1000 таңба қабылданады: {r.status_code}")

log("\n== Беттер")
for p in ["/", "/analyzer", "/how-it-works", "/performance", "/about"]:
    log(f"{p}: {c.get(p).status_code}")

Path(__file__).parent.joinpath("results", "test_run.txt").write_text("\n".join(out), encoding="utf-8")
sys.exit(1 if wrong else 0)
