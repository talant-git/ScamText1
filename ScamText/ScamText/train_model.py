"""Train TF-IDF + Logistic Regression on data/messages.xlsx and write real metrics."""
import json, pickle
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (log_loss, brier_score_loss, accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from text_utils import preprocess, strip_artifacts, translit

BASE = Path(__file__).parent
SEED = 42
LANG_MAP = {"English": "English", "Russian": "Russian", "Kazakh": "Kazakh"}


def make_vectorizer():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=1)


def make_model():
    return LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced", random_state=SEED)


def augment(df):
    """Add Latin-script (transliterated) copies of Kazakh/Russian TRAINING texts; same label. Never applied to test data."""
    cyr = df[df.Language.isin(["Kazakh", "Russian"])]
    parts = [df]
    for v in ("A", "B"):
        c = cyr.copy()
        c["clean"] = c.clean.map(lambda t, v=v: translit(t, v))
        parts.append(c)
    return pd.concat(parts, ignore_index=True)


def metrics(y_true, y_pred):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "support": int(len(y_true)),
    }


def main():
    base = pd.read_excel(BASE / "data" / "messages.xlsx", sheet_name="Dataset_3000")
    report = {"data_checks": {}}
    expected = ["Language", "Label", "Message", "Source_Attribution"]
    assert list(base.columns) == expected, f"Unexpected columns: {list(base.columns)}"
    report["data_checks"]["xlsx_rows"] = int(len(base))
    report["data_checks"]["null_values"] = int(base[expected].isna().sum().sum())
    report["data_checks"]["exact_duplicate_messages_in_xlsx"] = int(base.Message.duplicated().sum())
    assert set(base.Label.unique()) == {"Spam", "Ham (Normal)"}
    # Per the author's statement the xlsx messages were generated with Gemini (synthetic); label sources accordingly.
    base["Source_Attribution"] = "Gemini арқылы жасалған синтетикалық мәтін (messages.xlsx)"
    base["License"] = "Үшінші тарап лицензиясы жоқ (синтетикалық)"
    base["Origin"] = "synthetic"
    base["y"] = (base.Label.str.strip().str.lower() == "spam").astype(int)
    base["clean"] = base.Message.map(strip_artifacts)

    # --- leakage demonstration on the supplied xlsx: label-encoding suffixes (ID-n)/[Код: n] NOT removed
    tr, te = train_test_split(base, test_size=0.2, random_state=SEED, stratify=base.y)
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
    m = make_model().fit(v.fit_transform(tr.Message.str.lower()), tr.y)
    leak_acc = accuracy_score(te.y, m.predict(v.transform(te.Message.str.lower())))
    report["leakage_demo"] = {
        "description": "Accuracy if raw xlsx messages (with (ID-n)/[Код: n] suffixes that perfectly encode the label) are used. NOT a valid result.",
        "accuracy_with_label_leakage": round(float(leak_acc), 4),
    }

    # --- optional open-licensed data (created by add_open_data.py)
    raw = base
    open_sources = []
    extra_files = [f for f in [BASE / "data" / "open_data_uci.csv", BASE / "data" / "user_docx_data.csv", BASE / "data" / "kazakh_gemini_300.csv", BASE / "data" / "claude_ru_en.csv", BASE / "data" / "claude_zh.csv"] if f.exists()]
    extras = []
    for f in extra_files:
        e = pd.read_csv(f, encoding="utf-8").dropna(subset=["Message"])
        assert set(e.Label.unique()) <= {"Spam", "Ham (Normal)"}
        for col, dflt in (("License", "unknown"), ("Origin", "unknown")):
            if col not in e:
                e[col] = dflt
        e["y"] = (e.Label == "Spam").astype(int)
        e["clean"] = e.Message.map(strip_artifacts)
        extras.append(e)
        report["data_checks"]["rows_from_" + f.stem] = int(len(e))
    extra = pd.concat(extras, ignore_index=True) if extras else None
    if extra is not None:
        raw = pd.concat([base, extra[base.columns]], ignore_index=True)
    raw["key"] = raw.clean.map(preprocess)   # normalised text: numbers/dates/emoji removed, so template variants collapse
    report["data_checks"]["raw_rows"] = int(len(raw))

    # --- cleaning + deduplication
    before = len(raw)
    ds = raw.drop_duplicates(subset=["key", "Language", "y"]).copy()
    conflict_texts = ds.groupby("key").y.nunique()
    conflict_texts = set(conflict_texts[conflict_texts > 1].index)
    report["data_checks"]["texts_with_conflicting_labels_removed"] = len(conflict_texts)
    ds = ds[~ds.key.isin(conflict_texts)].copy()
    report["data_checks"]["unique_texts_after_cleaning"] = int(len(ds))
    report["data_checks"]["removed_duplicates"] = int(before - len(ds))
    for (src, lic, org, lang), g in ds.groupby(["Source_Attribution", "License", "Origin", "Language"]):
        open_sources.append({"source": src, "license": lic, "origin": org, "language": lang,
                             "unique_texts_used": int(len(g))})
    ds["strat"] = ds.Language + "_" + ds.y.astype(str)

    # Chinese is split separately, so the split of the other languages is exactly the same as before Chinese was added
    ds_main = ds[ds.Language != "Chinese"]
    train, test = train_test_split(ds_main, test_size=0.2, random_state=SEED, stratify=ds_main.strat)
    zh = ds[ds.Language == "Chinese"]
    if len(zh):
        zh_tr, zh_te = train_test_split(zh, test_size=0.2, random_state=SEED, stratify=zh.y)
        train = pd.concat([train, zh_tr]); test = pd.concat([test, zh_te])
    report["dataset"] = {
        "total_rows_in_file": int(before),
        "extra_sources": open_sources,
        "unique_texts_used": int(len(ds)),
        "train_size": int(len(train)),
        "test_size": int(len(test)),
        "split": "80/20 stratified by language and class, split done AFTER deduplication (no text appears in both sets)",
        "per_language_unique": ds.Language.value_counts().to_dict(),
        "per_language_class": {f"{l}": ds[ds.Language == l].Label.value_counts().to_dict() for l in ds.Language.unique()},
        "sources": {k: v for k, v in raw.groupby("Language").Source_Attribution.first().items()},
        "algorithm": "TF-IDF (char_wb n-grams 2-5, sublinear tf) + Logistic Regression (C=10)",
    }

    vec = make_vectorizer()
    train_aug = augment(train)
    X_tr = vec.fit_transform(train_aug.clean.map(preprocess))
    clf = make_model().fit(X_tr, train_aug.y)
    X_te = vec.transform(test.clean.map(preprocess))
    pred = clf.predict(X_te)
    report["overall"] = metrics(test.y, pred)
    cm = confusion_matrix(test.y, pred, labels=[0, 1])
    report["confusion_matrix"] = {"labels": ["Ham", "Spam"], "matrix": cm.tolist()}

    report["per_language"] = {}
    for lang in sorted(test.Language.unique()):
        mask = (test.Language == lang).values
        report["per_language"][lang] = metrics(test.y.values[mask], pred[mask])

    # --- same held-out KK/RU test texts, but written in Latin script (unseen transliteration variant mix)
    tt = test[test.Language.isin(["Kazakh", "Russian"])]
    if len(tt):
        tp = clf.predict(vec.transform(tt.clean.map(lambda t: translit(t, "A")).map(preprocess)))
        report["latin_script_test"] = metrics(tt.y, tp)

    # --- tougher check: 5-fold CV on the unique texts (more stable than a tiny 18-item test set)
    from sklearn.model_selection import StratifiedKFold
    accs = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(ds, ds.strat):
        va = make_vectorizer(); mm = make_model()
        fa = augment(ds.iloc[a])
        mm.fit(va.fit_transform(fa.clean.map(preprocess)), fa.y)
        accs.append(accuracy_score(ds.iloc[b].y, mm.predict(va.transform(ds.iloc[b].clean.map(preprocess)))))
    report["cross_validation_5fold_accuracy"] = {"mean": round(float(np.mean(accs)), 4), "folds": [round(float(x), 4) for x in accs]}

    # --- stronger test of generalization: leave-one-language-out is not meaningful for different scripts,
    # so we report only the per-language results above.

    # --- comparison with alternative methods: same split, same preprocessing, same training-only augmentation.
    # The chosen method (first row) was fixed beforehand; it is NOT re-selected from this table.
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC
    candidates = [
        ("TF-IDF (таңба n-грам 2-5) + Logistic Regression [таңдалған]", make_vectorizer, make_model),
        ("TF-IDF (сөз 1-2) + Logistic Regression",
         lambda: TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b\w+\b"), make_model),
        ("TF-IDF (таңба n-грам 2-5) + Naive Bayes", make_vectorizer, lambda: MultinomialNB(alpha=0.1)),
        ("TF-IDF (таңба n-грам 2-5) + Linear SVM", make_vectorizer,
         lambda: LinearSVC(C=1.0, class_weight="balanced", random_state=SEED)),
    ]
    comparison = []
    for name, mk_vec, mk_clf in candidates:
        v = mk_vec(); mdl = mk_clf()
        mdl.fit(v.fit_transform(train_aug.clean.map(preprocess)), train_aug.y)
        Xt = v.transform(test.clean.map(preprocess))
        pr = mdl.predict(Xt)
        row = {"model": name, **metrics(test.y, pr)}
        if hasattr(mdl, "predict_proba"):   # probability quality (lower is better); LinearSVC has no probabilities
            pp = mdl.predict_proba(Xt)[:, 1]
            row["log_loss"] = round(float(log_loss(test.y, pp, labels=[0, 1])), 4)
            row["brier"] = round(float(brier_score_loss(test.y, pp)), 4)
        row["per_language_accuracy"] = {
            lang: round(float(accuracy_score(test.y.values[(test.Language == lang).values], pr[(test.Language == lang).values])), 4)
            for lang in sorted(test.Language.unique())}
        cv = []
        for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(ds, ds.strat):
            fa = augment(ds.iloc[a]); vv = mk_vec(); mm2 = mk_clf()
            mm2.fit(vv.fit_transform(fa.clean.map(preprocess)), fa.y)
            cv.append(accuracy_score(ds.iloc[b].y, mm2.predict(vv.transform(ds.iloc[b].clean.map(preprocess)))))
        row["cv_accuracy_mean"] = round(float(np.mean(cv)), 4)
        comparison.append(row)
    report["model_comparison"] = comparison

    cnt = ds.Language.value_counts().to_dict()
    small = {k: v for k, v in cnt.items() if v < 200}
    lim = []
    if small:
        lim.append("Бірегей мәтін саны аз тілдер: " + ", ".join(f"{k} — {v}" for k, v in small.items())
                   + ". Олар үшін тест нәтижелері тұрақсыз және оптимистік болуы мүмкін.")
    real = ds[ds.Origin == "real"]
    if len(real) < len(ds):
        lim.append("Деректердің синтетикалық бөлігі (" + str(len(ds) - len(real)) + " бірегей мәтін) жасанды интеллект (Gemini, Claude) арқылы жасалған; бұл шынайы SMS емес, сондықтан нақты хабарламаларда нәтиже төмен болуы мүмкін.")
    if len(real):
        lim.append("Шынайы ашық дерек: " + str(len(real)) + " бірегей мәтін; олардың дереккөзі мен лицензиясы «Жоба туралы» бетінде көрсетілген.")
    lim.append("Деректер тілдер бойынша тең емес: " + ", ".join(f"{k} — {v}" for k, v in cnt.items()) + ". Тілдер арасындағы нәтижелерді тікелей салыстыруға болмайды.")
    lim.append("Транслит (латын әрпімен жазылған қазақша/орысша) қолдауы тек өз оқыту мәтіндерінің автоматты транслитерациясына негізделген, шынайы транслит корпусы жоқ.")
    lim.append("Spam деп анықталуы хабарламаның алаяқтық екенін дәлелдемейді; платформа тек мәтінді талдайды.")
    report["limitations"] = lim

    # QAIRU AI Fundamentals Final: extra from-scratch models (6-апта Bayes, 9-апта NN, 10-апта safety).
    # Adds new keys to `report` and new files; nothing above is modified.
    from extra_training import train_extra
    train_extra(BASE, train, test, train_aug, vec, clf, report, augment, make_vectorizer)

    with open(BASE / "models" / "vectorizer.pkl", "wb") as f:
        pickle.dump(vec, f)
    with open(BASE / "models" / "scamtext_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    (BASE / "results" / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Purples")
    ax.set_xticks([0, 1], ["Ham", "Spam"]); ax.set_yticks([0, 1], ["Ham", "Spam"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix (test set)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(BASE / "results" / "confusion_matrix.png", dpi=150)
    print(json.dumps({k: report[k] for k in ["data_checks", "leakage_demo", "overall", "per_language", "cross_validation_5fold_accuracy"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
