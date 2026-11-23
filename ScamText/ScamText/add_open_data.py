"""Download the UCI SMS Spam Collection (CC BY 4.0, 5,574 real English SMS) into data/open_data_uci.csv.

Usage:
    python add_open_data.py                  # downloads from UCI
    python add_open_data.py --file x.zip     # use a manually downloaded zip (or the raw SMSSpamCollection file)
Then run:  python train_model.py
"""
import argparse, hashlib, io, urllib.request, zipfile
from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent
OUT = BASE / "data" / "open_data_uci.csv"
URLS = [
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    "http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",
]
SOURCE = ("UCI Machine Learning Repository — SMS Spam Collection "
          "(Almeida & Gomez Hidalgo, 2011), https://doi.org/10.24432/C5CC84")
LICENSE = "CC BY 4.0"


def download():
    last = None
    for url in URLS:
        try:
            print("Жүктелуде:", url)
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            print(f"  {len(data)} байт, sha256={hashlib.sha256(data).hexdigest()[:16]}...")
            return data
        except Exception as e:  # noqa
            last = e
            print("  сәтсіз:", e)
    raise SystemExit(f"Жүктеу мүмкін болмады ({last}). UCI бетінен zip-ті қолмен жүктеп, "
                     "`python add_open_data.py --file файл.zip` деп іске қосыңыз.")


def read_text(raw: bytes) -> str:
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            name = next(n for n in z.namelist() if n.lower().endswith("smsspamcollection"))
            raw = z.read(name)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def parse(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        if "\t" not in line:
            continue
        label, msg = line.split("\t", 1)
        label, msg = label.strip().lower(), msg.strip()
        if label in ("ham", "spam") and msg:
            rows.append(("English", "Spam" if label == "spam" else "Ham (Normal)", msg))
    df = pd.DataFrame(rows, columns=["Language", "Label", "Message"])
    df["Source_Attribution"] = SOURCE
    df["License"] = LICENSE
    df["Origin"] = "real"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="қолмен жүктелген zip немесе SMSSpamCollection файлы")
    a = ap.parse_args()
    raw = Path(a.file).read_bytes() if a.file else download()
    df = parse(read_text(raw))
    print("Оқылды:", len(df), "жол;", df.Label.value_counts().to_dict())
    if len(df) != 5574:
        print("ЕСКЕРТУ: UCI бетінде 5574 жазба көрсетілген, мұнда басқа сан шықты. Файлды тексеріңіз.")
    df.to_csv(OUT, index=False, encoding="utf-8")
    print("Сақталды:", OUT, "\nЕнді іске қосыңыз: python train_model.py")


if __name__ == "__main__":
    main()
