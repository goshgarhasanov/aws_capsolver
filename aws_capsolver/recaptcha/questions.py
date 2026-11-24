"""Map a reCAPTCHA v2 question (Freebase MID or free text in many languages) to a class key.

``normalize_question("/m/0k4j")``                          -> "car"
``normalize_question("Select all images with traffic lights")`` -> "traffic_light"
``normalize_question("arabalar")``                         -> "car"
``normalize_question("светофоры")``                        -> "traffic_light"
``normalize_question("Select all images with guitars")``   -> "guitar"   (open vocabulary)

Known keys are listed in :data:`CLASSES`. Anything else that still looks like an English
object name is returned as a lower-case phrase and scored zero-shot by CLIP.

No heavy imports here: the service may call this for validation before the model loads.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from .errors import InvalidRecaptchaInput, UnsupportedQuestion

# Canonical class key -> English description used for CLIP prompts.
CLASSES: dict[str, str] = {
    "car": "car",
    "bus": "bus",
    "school_bus": "yellow school bus",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "truck": "truck",
    "tractor": "tractor",
    "taxi": "taxi",
    "boat": "boat",
    "traffic_light": "traffic light",
    "fire_hydrant": "fire hydrant",
    "crosswalk": "crosswalk",
    "parking_meter": "parking meter",
    "bridge": "bridge",
    "stairs": "stairs",
    "chimney": "chimney",
    "palm_tree": "palm tree",
    "mountain": "mountain or hill",
}

# Freebase MIDs sent by reCAPTCHA (and by CapSolver-style clients as ``question``).
# Verified against CapSolver's ReCaptchaV2Classification documentation and the Freebase /
# Google Knowledge Graph ids of the same names.
MIDS: dict[str, str] = {
    "/m/0k4j": "car",
    "/m/01bjv": "bus",
    "/m/02yvhj": "school_bus",
    "/m/04_sv": "motorcycle",
    "/m/0199g": "bicycle",
    "/m/07r04": "truck",
    "/m/013xlm": "tractor",
    "/m/0pg52": "taxi",
    "/m/019jd": "boat",
    "/m/015qff": "traffic_light",
    "/m/01pns0": "fire_hydrant",
    "/m/014xcs": "crosswalk",
    "/m/015qbp": "parking_meter",
    "/m/015kr": "bridge",
    "/m/01lynh": "stairs",
    "/m/01jk_4": "chimney",
    "/m/0cdl1": "palm_tree",
    "/m/09d_r": "mountain",
    "/m/05s2s": "plant",  # generic "plant"; not a regular reCAPTCHA class -> zero-shot
}

# Aliases per class. A trailing "*" marks a stem: up to 10 more letters may follow
# (inflected languages: tr/az/ru/uk/de/...). Without "*" the alias must match whole
# words, optionally followed by an English plural "s" / "es".
_ALIASES: dict[str, tuple[str, ...]] = {
    "car": (
        "car", "cars", "automobile", "vehicle", "vehicles",  # en
        "araba*", "otomobil*", "binek araç*",  # tr
        "avtomobil*", "maşın*", "masin*",  # az
        "автомобил*", "машин*", "легков*",  # ru / uk
        "автомобіл*",  # uk
        "coche*", "carro*", "auto", "autos", "automóvil*", "automovil*", "vehículo*",  # es / pt
        "pkw", "autos", "wagen",  # de
        "voiture*", "automobil*",  # fr
        "macchin*",  # it
        "samochod*", "samochód",  # pl
        "汽车", "轿车", "車", "자동차",  # zh / ja / ko
    ),
    "bus": (
        "bus", "buses", "busses", "otobüs*", "otobus*", "avtobus*", "автобус*", "autobús",
        "autobuses", "autobus", "ônibus", "onibus", "busse", "公共汽车", "公交车", "バス", "버스",
    ),
    "school_bus": (
        "school bus", "school buses", "okul otobüs*", "məktəb avtobus*", "школьн* автобус*",
        "autobús escolar", "autobuses escolares", "schulbus*", "bus scolaire", "校车",
    ),
    "motorcycle": (
        "motorcycle", "motorcycles", "motorbike", "motorbikes", "motor bike", "scooter",
        "motosiklet*", "мотоцикл*", "мопед*", "motocicleta*", "moto", "motos", "motorrad",
        "motorräder", "motorraeder", "motocyclette*", "motocicletta*", "motocykl*", "摩托车",
        "バイク", "오토바이",
    ),
    "bicycle": (
        "bicycle", "bicycles", "bike", "bikes", "bisiklet*", "velosiped*", "велосипед*",
        "bicicleta*", "bici", "fahrrad", "fahrräder", "fahrraeder", "vélo*", "velo", "velos",
        "bicyclette*", "bicicletta*", "biciclette", "rower*", "自行车", "自転車", "자전거",
    ),
    "truck": (
        "truck", "trucks", "lorry", "lorries", "kamyon*", "tır", "yük maşın*", "грузовик*",
        "вантажівк*", "camión", "camion", "camiones", "caminhão", "caminhões", "lkw",
        "lastwagen*", "lastkraftwagen*", "ciężarówk*", "卡车", "トラック", "트럭",
    ),
    "tractor": (
        "tractor", "tractors", "traktör*", "traktor*", "трактор*", "tractores", "trator*",
        "trecker*", "tracteur*", "trattor*", "拖拉机", "トラクター", "트랙터",
    ),
    "taxi": (
        "taxi", "taxis", "taxicab", "taxicabs", "cab", "cabs", "taksi*", "такси", "таксі",
        "táxi*", "出租车", "タクシー", "택시",
    ),
    "boat": (
        "boat", "boats", "ship", "ships", "tekne*", "gemi*", "qayıq*", "qayiq*", "gəmi*",
        "лодк*", "лодок", "катер*", "човен", "човн*", "barco*", "bote*", "boot", "boote",
        "schiff*", "bateau*", "barca", "barche", "łódź", "łodzie", "łodzi*", "船", "보트",
    ),
    "traffic_light": (
        "traffic light", "traffic lights", "traffic signal", "traffic signals", "stoplight",
        "stoplights", "trafik ışık*", "trafik isik*", "trafik lamba*", "svetofor*",
        "işıqfor*", "светофор*", "світлофор*", "semáforo*", "semaforo*", "semafor*",
        "ampel*", "verkehrsampel*", "feu de circulation", "feux de circulation",
        "feu tricolore", "feux tricolores", "feux de signalisation", "sygnalizacj*",
        "红绿灯", "交通灯", "信号機", "신호등",
    ),
    "fire_hydrant": (
        "fire hydrant", "fire hydrants", "hydrant", "hydrants", "yangın musluğ*",
        "yangin muslug*", "yangın musluk*", "yangın hidrant*", "hidrant*", "гидрант*",
        "пожарн* гидрант*", "пожежн* гідрант*", "гідрант*", "boca de incendio*",
        "bocas de incendio*", "hidrante*", "hydrant*", "bouche d'incendie",
        "bouches d'incendie", "borne d'incendie", "bornes d'incendie", "borne incendie",
        "idrant*", "消防栓", "消火栓", "소화전",
    ),
    "crosswalk": (
        "crosswalk", "crosswalks", "pedestrian crossing", "pedestrian crossings",
        "zebra crossing", "zebra crossings", "yaya geçid*", "yaya geçit*", "yaya gecid*",
        "yaya gecit*", "piyada keçid*", "piyada kecid*", "пешеходн* переход*", "зебр*",
        "пішохідн* перехід*", "пішохідн* переход*", "paso de peatones", "pasos de peatones",
        "paso peatonal", "pasos peatonales", "cruce peatonal", "cruces peatonales",
        "faixa de pedestre*", "faixas de pedestre*", "passadeira*", "zebrastreifen",
        "fußgängerüberweg*", "fussgaengerueberweg*", "passage piéton*", "passages piéton*",
        "passage pour piétons", "passages pour piétons", "strisce pedonali",
        "attraversament* pedonal*", "przejści* dla pieszych", "人行横道", "斑马线", "横断歩道",
        "횡단보도",
    ),
    "parking_meter": (
        "parking meter", "parking meters", "parkmetre*", "park sayac*", "park sayaç*",
        "parkomat*", "parkinq sayğac*", "паркомат*", "парковочн* счетчик*",
        "парковочн* счётчик*", "parquímetro*", "parquimetro*", "parkuhr*",
        "parkscheinautomat*", "parcmètre*", "parcmetre*", "parchimetr*", "停车计时器",
        "パーキングメーター", "주차 미터기",
    ),
    "bridge": (
        "bridge", "bridges", "köprü*", "kopru*", "körpü*", "korpu*", "мост*", "міст",
        "мости*", "мостів", "puente*", "ponte*", "brücke*", "bruecke*", "pont", "ponts",
        "ponti", "most", "mosty", "桥", "橋", "다리",
    ),
    "stairs": (
        "stairs", "stair", "staircase", "staircases", "stairway", "stairways", "steps",
        "merdiven*", "pilləkən*", "pilleken*", "лестниц*", "ступен*", "сходи", "сходинк*",
        "escalera*", "escada*", "treppe*", "escalier*", "scala", "scale", "scalinat*",
        "schody", "schodów", "楼梯", "階段", "계단",
    ),
    "chimney": (
        "chimney", "chimneys", "baca", "bacalar*", "bacaları", "bacası", "dymov*",
        "дымов* труб*", "дымоход*", "димар*", "димов* труб*", "chimenea*", "chaminé*",
        "chamine*", "schornstein*", "kamin*", "cheminée*", "cheminee*", "camino", "camini",
        "comignol*", "komin*", "烟囱", "煙突", "굴뚝",
    ),
    "palm_tree": (
        "palm tree", "palm trees", "palm", "palms", "palmiye*", "palm ağac*", "palma",
        "palmalar*", "пальм*", "palmera*", "palmeira*", "palme", "palmen", "palmier*",
        "棕榈树", "ヤシの木", "야자수",
    ),
    "mountain": (
        "mountain", "mountains", "hill", "hills", "mountains or hills", "dağ", "dağlar*",
        "dağları", "tepe", "tepeler*", "təpə*", "tepə*", "горы", "гор", "горами", "гора",
        "холм*", "гори", "пагорб*", "montaña*", "colina*", "montanha*", "berg", "berge",
        "hügel", "montagne*", "colline*", "montagn*", "collin*", "góry", "gór", "wzgórz*",
        "山", "언덕",
    ),
}

# Boilerplate words removed before the open-vocabulary fallback (English only).
_EN_LEAD = re.compile(
    r"^(?:please\s+)?(?:select|click|choose|pick|tap)\s+(?:on\s+)?(?:all|every|each|the)?\s*"
    r"(?:of\s+the\s+)?(?:images?|squares?|tiles?|pictures?|photos?|boxes?)?\s*"
    r"(?:with|containing|that\s+contain|that\s+have|showing|of|which\s+contain)?\s*")
_EN_TAIL = re.compile(r"\s*(?:\.|,|;|!|if there are none.*|click verify.*|then verify.*|once there are none.*)$")
_PHRASE_RE = re.compile(r"^[a-z][a-z' -]{1,39}$")
_STOP = {"a", "an", "the", "some", "any", "all", "and", "or", "of", "with", "images", "image",
         "squares", "square", "tiles", "tile", "pictures", "picture", "none", "left", "verify",
         "skip", "click", "select"}


def _fold(text: str) -> str:
    """Lower-case + NFKC + whitespace/punctuation normalisation (keeps letters of any script)."""
    # Turkish dotted capital I would casefold to "i" + combining dot; map it to plain "i".
    text = unicodedata.normalize("NFKC", text).replace("İ", "i")
    text = text.casefold().replace("i̇", "i")
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"[^\w' -]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _alias_regex(alias: str) -> tuple[re.Pattern, int]:
    stem = alias.endswith("*")
    body = _fold(alias.rstrip("*"))
    parts = [re.escape(p) + (r"\w{0,10}" if stem else "") for p in body.split(" ")]
    pat = r"\s+".join(parts)
    if not stem:
        pat += r"(?:s|es)?"
    # CJK has no word boundaries: plain substring match there.
    if re.search(r"[぀-ヿ㐀-鿿가-힯]", body):
        return re.compile(re.escape(body)), len(body) * 3
    return re.compile(rf"(?<!\w){pat}(?!\w)"), len(body)


@lru_cache(maxsize=1)
def _compiled() -> list[tuple[int, re.Pattern, str]]:
    out = []
    for key, aliases in _ALIASES.items():
        for a in aliases:
            rx, weight = _alias_regex(a)
            out.append((weight, rx, key))
    out.sort(key=lambda t: -t[0])  # longest (most specific) alias first
    return out


def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "sses", "xes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _open_vocab_phrase(folded: str) -> str | None:
    t = _EN_TAIL.sub("", _EN_LEAD.sub("", folded)).strip()
    t = re.sub(r"^(?:a|an|the|some)\s+", "", t)
    if not t or not _PHRASE_RE.match(t):
        return None
    words = [w for w in t.split(" ") if w]
    if not 1 <= len(words) <= 4 or all(w in _STOP for w in words):
        return None
    words[-1] = _singular(words[-1])
    return " ".join(words)


def normalize_question(question: object) -> str:
    """Return the canonical class key for a MID or text question.

    Known classes map to the keys of :data:`CLASSES`; other English object names are
    returned as a short lower-case phrase (scored zero-shot). Raises UnsupportedQuestion
    for unknown MIDs and for text that names nothing recognisable, InvalidRecaptchaInput
    for non-string input.
    """
    if not isinstance(question, str):
        raise InvalidRecaptchaInput(f"question must be a string, got {type(question).__name__}")
    q = question.strip()
    if not q:
        raise UnsupportedQuestion("empty question")
    if len(q) > 300:
        raise UnsupportedQuestion("question is too long")
    if q.startswith("/m/") or q.startswith("/g/"):
        key = MIDS.get(q.lower() if q.startswith("/m/") else q)
        if key is None:
            raise UnsupportedQuestion(f"unknown reCAPTCHA object id {q!r}")
        return key
    folded = _fold(q)
    if folded.replace(" ", "_") in CLASSES:
        return folded.replace(" ", "_")
    for _, rx, key in _compiled():
        if rx.search(folded):
            return key
    phrase = _open_vocab_phrase(folded)
    if phrase is None:
        raise UnsupportedQuestion(f"cannot understand reCAPTCHA question {question!r}")
    return phrase


def prompt_name(key: str) -> str:
    """English name of a class key for CLIP prompts ("traffic_light" -> "traffic light")."""
    return CLASSES.get(key, key.replace("_", " "))
