"""Browser test of every page and button (needs: pip install playwright && playwright install chromium).
Start the server first: python app.py. Writes results/ui_test.txt."""
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:5000"
out = []
def log(s):
    out.append(s); print(s)
def check(name, cond):
    log(("OK   " if cond else "ҚАТЕ ") + name)
    return cond

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1280, "height": 800})
    errors = []
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errors.append(str(e)))

    pg.goto(BASE)
    check("Басты бет ашылды, логотип бар", pg.locator("nav .logo").count() == 1 and "ScamText" in pg.title())
    check("«Хабарламаны тексеру» батырмасы бар", pg.get_by_role("link", name="Хабарламаны тексеру").count() == 1)
    check("Тілдер туралы блок бар (қазақша, орысша, ағылшынша, қытайша)", all(pg.get_by_text(t).count() >= 1 for t in ["Қазақша", "Русский", "English", "中文"]))
    check("Қауіпсіздік ескертуі бар", pg.get_by_text("Қауіпсіздік туралы").count() == 1)
    pg.get_by_role("link", name="Хабарламаны тексеру").click(); pg.wait_for_url("**/analyzer")
    check("Басты беттегі батырма анализаторға апарады", pg.url.endswith("/analyzer"))

    # empty input
    pg.click("#check")
    check("Бос мәтін: түсінікті қате көрсетіледі", pg.locator("#err").is_visible() and "бос" in pg.inner_text("#err"))
    check("Бос мәтін: нәтиже көрсетілмейді", not pg.locator("#result").is_visible())

    # three languages
    for name, text, exp in [("қазақша", "Сізге 200 000 теңге несие бекітілді! Ақшаны алу үшін сілтемеге өтіңіз.", "SPAM"),
                            ("орысша", "Привет, завтра в 10:00 созвонимся по проекту.", "HAM"),
                            ("ағылшынша", "WINNER! You have been selected for a free iPhone, click here to claim your prize now", "SPAM")]:
        pg.fill("#msg", text); pg.click("#check"); pg.wait_for_selector("#result", state="visible")
        v = pg.inner_text("#verdict")
        log(f"     {name}: нәтиже={v}, тіл={pg.inner_text('#rlang')}, сенімділік={pg.inner_text('#rconf')}")
        check(f"{name} мәтін талданды және нәтиже шықты (күтілген {exp})", v == exp and pg.inner_text('#rconf').endswith("%"))
    pg.fill("#msg", "恭喜您获得iPhone一台，点击链接填写收货信息并支付10元运费。"); pg.click("#check"); pg.wait_for_function("document.getElementById('rlang').textContent==='Chinese'")
    check("Қытайша мәтін талданды: тіл Chinese, нәтиже SPAM, түсіндірме сөздер шықты", pg.inner_text("#verdict") == "SPAM" and pg.locator("#rwords .chip").count() >= 1)
    check("Нәтижеде түсіндірме (сөздер) бөлімі бар", pg.locator("#rhint").inner_text().strip() != "")

    # manual language selection
    pg.select_option("#lang", "Russian"); pg.click("#check"); pg.wait_for_function("document.getElementById('rlang').textContent==='Russian'")
    check("Тілді қолмен таңдау нәтижеде көрінеді", pg.inner_text("#rlang") == "Russian")

    # re-check button
    pg.click("#again")
    check("«Қайта тексеру» өрісті тазалап, нәтижені жасырады", pg.input_value("#msg") == "" and not pg.locator("#result").is_visible())

    # too long
    pg.fill("#msg", "a" * 1001); pg.click("#check")
    check("Тым ұзын мәтін: қате көрсетіледі", pg.locator("#err").is_visible() and "ұзын" in pg.inner_text("#err"))
    check("Таңба санағышы жұмыс істейді", pg.inner_text("#cnt").startswith("1001"))

    # XSS
    pg.fill("#msg", "<img src=x onerror=window.__xss=1> free prize click now"); pg.click("#check"); pg.wait_for_selector("#result", state="visible")
    check("<img onerror> енгізілімі орындалмады (XSS жоқ)", pg.evaluate("window.__xss") is None)

    # server down message
    pg.route("**/api/analyze", lambda r: r.abort())
    pg.fill("#msg", "test message"); pg.click("#check"); pg.wait_for_selector("#err", state="visible")
    check("Сервер жауап бермесе, түсінікті қате көрсетіледі", "Серверге" in pg.inner_text("#err"))
    pg.unroute("**/api/analyze")

    # navigation + all pages
    for label, path, marker in [("Басты бет", "/", "Қолдау көрсетілетін тілдер"), ("Тексеру", "/analyzer", "Хабарламаны талдау"),
                                ("Қалай жұмыс істейді", "/how-it-works", "Жүйенің жұмыс принципі"),
                                ("Модель нәтижелері", "/performance", "Confusion Matrix"), ("Жоба туралы", "/about", "Шектеулер")]:
        pg.get_by_role("link", name=label, exact=True).first.click(); pg.wait_for_load_state()
        check(f"Навигация «{label}» → {path}", pg.url.endswith(path) and pg.get_by_text(marker).count() >= 1)

    pg.goto(BASE + "/how-it-works")
    check("How It Works: 5 қадам көрсетілген", pg.locator(".step").count() == 5)
    pg.goto(BASE + "/performance")
    t = pg.inner_text("main")
    check("Performance: Accuracy/Precision/Recall/F1 бар", all(k in t for k in ["Accuracy", "Precision", "Recall", "F1-score"]))
    check("Performance: Confusion Matrix, Train/Test, тіл кестесі, шектеулер бар", all(k in t for k in ["Confusion Matrix", "Train / Test", "Тіл бойынша нәтиже", "Модельдің шектеулері"]))
    pg.goto(BASE + "/about")
    t = pg.inner_text("main")
    check("About: технологиялар, мүмкіндіктер, дереккөз/лицензия, шектеулер бар", all(k in t for k in ["Технологиялар", "Мүмкіндіктер", "Дереккөздер", "Шектеулер"]))

    # mobile
    m = b.new_page(viewport={"width": 375, "height": 740})
    for path in ["/", "/analyzer", "/how-it-works", "/performance", "/about"]:
        m.goto(BASE + path)
        check(f"Телефон 375px: {path} көлденең айналдыру жоқ", m.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"))
    m.goto(BASE + "/"); m.screenshot(path=str(Path(__file__).parent / "results" / "screenshot_mobile.png"))
    pg.goto(BASE + "/"); pg.screenshot(path=str(Path(__file__).parent / "results" / "screenshot_home.png"))
    real = [e for e in errors if "ERR_FAILED" not in e]   # ERR_FAILED comes from the test's own simulated server outage
    check("Консольде JavaScript қатесі жоқ (өз үзілісімізден басқа)", not real)
    if real: log("     " + "; ".join(real[:3]))
    b.close()

fails = sum(1 for l in out if l.startswith("ҚАТЕ"))
log(f"\nҚате саны: {fails}")
Path(__file__).parent.joinpath("results", "ui_test.txt").write_text("\n".join(out), encoding="utf-8")
