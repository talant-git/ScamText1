"""Shared text preprocessing: used by BOTH training and inference."""
import re

# Artifacts present in the supplied dataset that directly encode the label
# (Ham rows end with "(ID-123)", Spam rows end with "[Код: 1234-56]").
_ID_SUFFIX = re.compile(r"\s*\(ID-\d+\)\s*$", re.IGNORECASE)
_CODE_SUFFIX = re.compile(r"\s*\[(?:Код|Code|Kod):\s*[\d\-]+\]\s*$", re.IGNORECASE)
_URL = re.compile(r"(https?://\S+|www\.\S+|\b\w+\.(?:com|kz|ru|org|net)\S*|wa\.me/\S+)", re.IGNORECASE)
_DATE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
_TIME = re.compile(r"\b\d{1,2}:\d{2}\b")
_NUM = re.compile(r"\d+")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2190-\u21FF\u2300-\u23FF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200d]")
_SPACES = re.compile(r"\s+")

KAZ_LETTERS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")


def strip_artifacts(text: str) -> str:
    text = _ID_SUFFIX.sub("", text)
    text = _CODE_SUFFIX.sub("", text)
    return text.strip()


def preprocess(text: str) -> str:
    text = strip_artifacts(str(text))
    text = _EMOJI.sub(" ", text)  # emoji removed: in the supplied data they correlate perfectly with Spam (artifact)
    text = text.lower()
    text = _URL.sub(" urltoken ", text)
    text = _DATE.sub(" datetoken ", text)
    text = _TIME.sub(" timetoken ", text)
    text = _NUM.sub(" numtoken ", text)
    text = _SPACES.sub(" ", text).strip()
    return text


def detect_language(text: str) -> str:
    """Script-based heuristic: Kazakh-specific letters -> Kazakh, other Cyrillic -> Russian, Latin -> English."""
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf")
    cyr = sum(1 for c in text if "\u0400" <= c <= "\u04FF")
    lat = sum(1 for c in text if c.isascii() and c.isalpha())
    if cjk > 0 and cjk >= cyr and cjk * 3 >= lat:
        return "Chinese"
    if cyr == 0 and lat == 0:
        return "Unknown"
    if lat > cyr:
        return "English"
    kaz = sum(1 for c in text if c in KAZ_LETTERS)
    # letters in the Kazakh-only set; one or two are enough since they are very frequent in Kazakh
    if kaz >= 1:
        return "Kazakh"
    return "Russian"


# --- Cyrillic -> Latin transliteration (used ONLY to augment training data from the user's own Kazakh/Russian texts)
_BASE = {"а":"a","б":"b","в":"v","г":"g","д":"d","е":"e","ё":"yo","ж":"zh","з":"z","и":"i","к":"k","л":"l","м":"m",
         "н":"n","о":"o","п":"p","р":"r","с":"s","т":"t","у":"u","ф":"f","ц":"ts","ч":"ch","ш":"sh","щ":"sh","ъ":"",
         "ы":"y","ь":"","э":"e","ю":"yu","я":"ya","ә":"a","ғ":"g","ң":"n","ө":"o","ұ":"u","ү":"u","һ":"h","і":"i"}
_VARIANTS = {
    "A": {"й":"y","х":"kh","қ":"q"},
    "B": {"й":"i","х":"h","қ":"k"},
}


def translit(text: str, variant: str = "A") -> str:
    m = {**_BASE, **_VARIANTS[variant]}
    out = []
    for ch in text:
        lo = ch.lower()
        if lo in m:
            t = m[lo]
            out.append(t.capitalize() if ch != lo else t)
        else:
            out.append(ch)
    return "".join(out)
