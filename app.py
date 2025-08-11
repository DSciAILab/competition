# ================================
# app.py – ADSS Jiu-Jitsu Competition
# + Aba Events (admin) para cadastro/gestão de eventos
# + Seleção de evento ativo antes do menu principal
# + Todas as ações vinculadas ao evento ativo (event_id)
# ================================

# ---------- IMPORTS ----------
import os, re, uuid, urllib.parse, base64, datetime as dt
from pathlib import Path
from io import BytesIO

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from PIL import Image, ImageDraw, ImageFont

import numpy as np
try:
    import cv2
    FACE_DETECT_AVAILABLE = True
except Exception:
    cv2 = None
    FACE_DETECT_AVAILABLE = False

import imagehash

# streamlit-webrtc (opcional com fallback + stubs p/ evitar NameError)
WEBRTC_AVAILABLE = True
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
except Exception:
    WEBRTC_AVAILABLE = False
    # Stubs para evitar NameError quando o pacote não está disponível
    webrtc_streamer = None
    WebRtcMode = None
    RTCConfiguration = None
    class VideoProcessorBase:  # stub
        pass

# ====== OCR (Google Vision) opcional ======
USE_VISION_OCR = False
try:
    if "USE_VISION_OCR" in st.secrets:
        USE_VISION_OCR = str(st.secrets["USE_VISION_OCR"]).strip().lower() in ("1","true","yes","on")
except Exception:
    pass
if not USE_VISION_OCR:
    USE_VISION_OCR = str(os.getenv("USE_VISION_OCR","")).strip().lower() in ("1","true","yes","on")

if USE_VISION_OCR:
    try:
        from google.cloud import vision
        if "GCP_CREDS_B64" in st.secrets:
            creds_b64 = st.secrets["GCP_CREDS_B64"]
            cred_path = "gcp_creds.json"
            with open(cred_path, "w") as f:
                f.write(base64.b64decode(creds_b64.encode()).decode())
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
        vision_client = vision.ImageAnnotatorClient()
    except Exception as e:
        USE_VISION_OCR = False
        st.warning(f"OCR via Google Vision não pôde ser inicializado: {e}. Usando fallback local.")

# ================================
# 1) CONFIG GLOBAL
# ================================
st.set_page_config(page_title="ADSS Jiu-Jitsu Competition", layout="wide")
st.markdown("""
<style>
#MainMenu, header, footer { visibility: hidden; }
.block-container { padding-top: .6rem; padding-bottom: 3rem; max-width: 860px; }
.stButton > button { width: 100%; padding: 1rem 1.1rem; border-radius: 14px; font-weight: 700; text-align: left; white-space: pre-wrap; line-height: 1.25; }
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] { min-height: 44px; }
[data-testid="stDataFrame"] { border-radius: 12px; }
.header-sub { color:#9aa0a6; margin:.25rem 0 1rem; text-align:center; }
.section-gap { height: .6rem; }
.badge-pending{display:inline-block;padding:2px 8px;border-radius:999px;background:#F59E0B;color:#111;font-size:12px;font-weight:600;margin-left:8px;}
.badge-approved{display:inline-block;padding:2px 8px;border-radius:999px;background:#10B981;color:#111;font-size:12px;font-weight:600;margin-left:8px;}
.preview-box { border:1px solid #2a2a2a; border-radius:12px; padding:12px; background:#0f1116; }
.card { border:1px solid #2b2b2b; border-radius:12px; padding:12px; margin-bottom:10px; display:flex; gap:12px; align-items:flex-start; }
.card img { border-radius:8px; }
.card .meta { font-size:0.95rem; line-height:1.3; }
.group-title { margin-top:18px; padding:8px 10px; background:#0f1116; border:1px solid #2b2b2b; border-radius:10px; font-weight:700; }
.help { color:#9aa0a6; font-size:.88rem; margin-top:.25rem;}
</style>
""", unsafe_allow_html=True)

# ================================
# 2) SECRETS / ENV / PASTAS
# ================================
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", None)
try:
    if "ADMIN_PASSWORD" in st.secrets:
        ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
except Exception:
    pass
if not ADMIN_PASSWORD: ADMIN_PASSWORD = "admin123"
WHATSAPP_PHONE = os.getenv("WHATSAPP_ORG", "+9715xxxxxxxx")

DB_PATH = "registrations.db"
UPLOAD_DIR = Path("uploads"); UPLOAD_DIR.mkdir(exist_ok=True)

EVENT_DEFAULT_REGION = "Al Dhafra – Abu Dhabi"

# ================================
# 3) BANCO DE DADOS (SQLite)
# ================================
engine = create_engine(f"sqlite:///{DB_PATH}", future=True)

def init_db():
    with engine.begin() as conn:
        # Tabela de inscrições
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS registrations (
            id TEXT PRIMARY KEY,
            created_at TEXT,
            event_id TEXT,               -- <- novo: vínculo com events.id
            event_name TEXT,             -- legibilidade
            first_name TEXT,
            last_name TEXT,
            full_name_en TEXT,
            email TEXT,
            phone TEXT,
            eid TEXT,
            nationality TEXT,
            gender TEXT,
            dob TEXT,
            age_years INTEGER,
            age_division TEXT,
            academy TEXT,
            coach TEXT,
            coach_phone TEXT,
            region TEXT,
            modality TEXT,
            belt TEXT,
            weight_class TEXT,
            weight_kg REAL,
            category TEXT,
            consent INTEGER,
            profile_photo_path TEXT,
            id_doc_photo_path TEXT,
            approval_status TEXT,
            face_phash TEXT,
            guardian_name TEXT,
            guardian_eid TEXT,
            guardian_phone TEXT,
            guardian_eid_photo_path TEXT
        );
        """))
        # Tabela de eventos
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            created_at TEXT,
            name TEXT,
            date TEXT,            -- ISO date (yyyy-mm-dd) ou datetime
            location TEXT,
            manager TEXT,         -- responsável
            divisions TEXT,       -- texto livre (divisões ativas)
            description TEXT,
            status TEXT           -- 'Ativo' ou 'Inativo'
        );
        """))
        # Migrações tolerantes
        for alter in [
            "ALTER TABLE registrations ADD COLUMN eid TEXT",
            "ALTER TABLE registrations ADD COLUMN weight_kg REAL",
            "ALTER TABLE registrations ADD COLUMN guardian_name TEXT",
            "ALTER TABLE registrations ADD COLUMN guardian_eid TEXT",
            "ALTER TABLE registrations ADD COLUMN guardian_phone TEXT",
            "ALTER TABLE registrations ADD COLUMN guardian_eid_photo_path TEXT",
            "ALTER TABLE registrations ADD COLUMN first_name TEXT",
            "ALTER TABLE registrations ADD COLUMN last_name TEXT",
            "ALTER TABLE registrations ADD COLUMN full_name_en TEXT",
            "ALTER TABLE registrations ADD COLUMN coach_phone TEXT",
            "ALTER TABLE registrations ADD COLUMN age_years INTEGER",
            "ALTER TABLE registrations ADD COLUMN approval_status TEXT",
            "ALTER TABLE registrations ADD COLUMN face_phash TEXT",
            "ALTER TABLE registrations ADD COLUMN event_id TEXT",
            "ALTER TABLE registrations ADD COLUMN event_name TEXT"
        ]:
            try: conn.execute(text(alter))
            except Exception: pass
init_db()

# ---------- REGISTRATIONS DAO ----------
def insert_registration(row: dict):
    with engine.begin() as conn:
        cols = ",".join(row.keys()); vals = ",".join([f":{k}" for k in row.keys()])
        conn.execute(text(f"INSERT INTO registrations ({cols}) VALUES ({vals})"), row)

def update_registration(reg_id: str, updates: dict):
    placeholders = ", ".join([f"{k} = :{k}" for k in updates.keys()])
    updates["id"] = reg_id
    with engine.begin() as conn:
        conn.execute(text(f"UPDATE registrations SET {placeholders} WHERE id = :id"), updates)

def fetch_all(event_id: str|None=None):
    with engine.begin() as conn:
        if event_id:
            return pd.read_sql("SELECT * FROM registrations WHERE event_id = :e ORDER BY created_at DESC", conn, params={"e": event_id})
        return pd.read_sql("SELECT * FROM registrations ORDER BY created_at DESC", conn)

def fetch_distinct_academies(event_id: str|None=None):
    with engine.begin() as conn:
        try:
            if event_id:
                rows = conn.execute(text("SELECT DISTINCT academy FROM registrations WHERE event_id=:e AND academy IS NOT NULL AND academy <> ''"), {"e":event_id}).fetchall()
            else:
                rows = conn.execute(text("SELECT DISTINCT academy FROM registrations WHERE academy IS NOT NULL AND academy <> ''")).fetchall()
            return sorted({r[0] for r in rows})
        except Exception:
            return []

def count_by_category(category_value: str, event_id: str|None=None):
    with engine.begin() as conn:
        try:
            if event_id:
                row = conn.execute(text("SELECT COUNT(*) FROM registrations WHERE event_id=:e AND category = :c"), {"e":event_id,"c":category_value}).fetchone()
            else:
                row = conn.execute(text("SELECT COUNT(*) FROM registrations WHERE category = :c"), {"c":category_value}).fetchone()
            return int(row[0] if row else 0)
        except Exception:
            return 0

# ---------- EVENTS DAO ----------
def insert_event(row: dict):
    with engine.begin() as conn:
        cols = ",".join(row.keys()); vals = ",".join([f":{k}" for k in row.keys()])
        conn.execute(text(f"INSERT INTO events ({cols}) VALUES ({vals})"), row)

def update_event(event_id: str, updates: dict):
    placeholders = ", ".join([f"{k} = :{k}" for k in updates.keys()])
    updates["id"] = event_id
    with engine.begin() as conn:
        conn.execute(text(f"UPDATE events SET {placeholders} WHERE id = :id"), updates)

def fetch_events(status: str|None=None):
    with engine.begin() as conn:
        if status:
            return pd.read_sql("SELECT * FROM events WHERE status = :s ORDER BY date ASC, created_at DESC", conn, params={"s":status})
        return pd.read_sql("SELECT * FROM events ORDER BY date ASC, created_at DESC", conn)

def get_event(event_id: str):
    with engine.begin() as conn:
        df = pd.read_sql("SELECT * FROM events WHERE id = :i", conn, params={"i":event_id})
        return None if df.empty else df.iloc[0].to_dict()

# ================================
# 4) PIL helper (textbbox compat)
# ================================
def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    bbox = draw.textbbox((0,0), text, font=font)
    return (bbox[2]-bbox[0], bbox[3]-bbox[1])

# ================================
# 5) IMAGEM / FACE / PHASH / CARDS
# ================================
def _file_to_pil(file_or_pil):
    if file_or_pil is None: return None
    if isinstance(file_or_pil, Image.Image): return file_or_pil
    try:
        return Image.open(BytesIO(file_or_pil.getvalue())).convert("RGB")
    except Exception:
        try:
            return Image.open(file_or_pil).convert("RGB")
        except Exception:
            return None

def save_image(file_or_pil, reg_id: str, suffix: str, max_size=(800, 800)) -> str:
    img = _file_to_pil(file_or_pil)
    if img is None: return ""
    img.thumbnail(max_size)
    dest = UPLOAD_DIR / f"{reg_id}_{suffix}.jpg"
    img.save(dest, format="JPEG", quality=85)
    return str(dest)

def count_faces_in_image(file_or_pil) -> int:
    if not FACE_DETECT_AVAILABLE: return -1
    img = _file_to_pil(file_or_pil)
    if img is None: return 0
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(80,80))
    return 0 if faces is None else len(faces)

def crop_face_from_uploaded(file_or_pil):
    img = _file_to_pil(file_or_pil)
    if img is None: return None
    if FACE_DETECT_AVAILABLE:
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(80,80))
        if faces is not None and len(faces)>0:
            x,y,w,h = max(faces, key=lambda r:r[2]*r[3])
            return img.crop((x,y,x+w,y+h))
    return img

def compute_phash(img_pil: Image.Image) -> str:
    if img_pil is None: return ""
    return str(imagehash.phash(img_pil))

def phash_distance(a: str, b: str) -> int:
    if not a or not b: return 1_000_000
    return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)

def generate_registration_jpg(row: dict, event_name: str, region: str, logo_file=None, width=1080, height=1350) -> bytes:
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 60)
        font_sub   = ImageFont.truetype("DejaVuSans.ttf", 32)
        font_bold  = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
        font_text  = ImageFont.truetype("DejaVuSans.ttf", 34)
    except Exception:
        font_title = font_sub = font_bold = font_text = ImageFont.load_default()

    y = 40
    if logo_file is not None:
        try:
            lf = Image.open(logo_file).convert("RGBA")
            maxw = int(width * 0.5); ratio = maxw / lf.width
            lf = lf.resize((maxw, int(lf.height * ratio)))
            img.paste(lf, (int((width - lf.width) / 2), y), lf); y += lf.height + 20
        except Exception:
            pass

    title_text = event_name or "ADSS Jiu-Jitsu Competition"
    w_title, _ = text_size(draw, title_text, font_title)
    draw.text(((width - w_title) / 2, y), title_text, fill=(0, 0, 0), font=font_title); y += 70
    if region:
        w_reg, _ = text_size(draw, region, font_sub)
        draw.text(((width - w_reg) / 2, y), region, fill=(80, 80, 80), font=font_sub); y += 50
    draw.line([(80, y), (width - 80, y)], fill=(230,230,230), width=3); y += 30

    x_left = 80; photo_size = 320
    profile_path = row.get("profile_photo_path", "")
    if profile_path and os.path.exists(profile_path):
        try:
            p = Image.open(profile_path).convert("RGB").resize((photo_size, photo_size))
            img.paste(p, (x_left, y))
        except Exception:
            pass

    x_text = x_left + photo_size + 40
    def line(label, value):
        nonlocal y
        draw.text((x_text, y), label, fill=(0,0,0), font=font_bold); y += 40
        draw.text((x_text, y), value, fill=(40,40,40), font=font_text); y += 54

    name = f"{row.get('first_name','')} {row.get('last_name','')}".strip()
    line("Inscrição (ID):", row.get("id",""))
    line("Nome:", name or "-")
    line("EID:", row.get("eid","-") or "-")
    line("Categoria:", row.get("category","-") or "-")
    line("Faixa:", row.get("belt","-") or "-")
    wkg = row.get("weight_kg", None)
    line("Peso atual:", f"{wkg:.1f} kg" if isinstance(wkg,(int,float)) else "-")
    line("Nacionalidade:", row.get("nationality","-") or "-")

    y = max(y, 80 + photo_size + 30)
    draw.line([(80, y), (width - 80, y)], fill=(230,230,230), width=3); y += 30

    col_x = 80
    def kv(k, v):
        nonlocal y
        draw.text((col_x, y), k, fill=(0,0,0), font=font_bold); y += 40
        draw.text((col_x, y), v, fill=(40,40,40), font=font_text); y += 52
    kv("E-mail:", row.get("email","-") or "-")
    kv("Telefone:", row.get("phone","-") or "-")
    kv("Professor:", row.get("coach","-") or "-")
    kv("Status:", row.get("approval_status","Pending"))

    try:
        import qrcode
        qr_data = f"Inscrição: {row.get('id','')}\nEvento: {event_name}"
        qr_img = qrcode.make(qr_data).resize((240, 240))
        img.paste(qr_img, (width - 80 - 240, y - 280))
    except Exception:
        pass

    buf = BytesIO(); img.save(buf, format="JPEG", quality=90, optimize=True); return buf.getvalue()

def generate_membership_card(img_profile_path: str, membership_id: str, eid: str, age: int, name: str, belt: str, width=900, height=600) -> bytes:
    canvas = Image.new("RGB", (width, height), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([(10,10),(width-10,height-10)], outline=(200,200,200), width=3)
    try:
        f_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 42)
        f_text  = ImageFont.truetype("DejaVuSans.ttf", 28)
        f_bold  = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
    except Exception:
        f_title = f_text = f_bold = ImageFont.load_default()

    x_photo, y_photo, psize = 30, 70, 220
    if img_profile_path and os.path.exists(img_profile_path):
        try:
            p = Image.open(img_profile_path).convert("RGB").resize((psize, psize))
            canvas.paste(p, (x_photo, y_photo))
        except Exception:
            pass

    draw.text((30, 20), "ADSS Jiu-Jitsu – Membership", fill=(0,0,0), font=f_title)

    x = x_photo + psize + 30; y = 70
    def field(k,v):
        nonlocal y
        draw.text((x, y), k, fill=(0,0,0), font=f_bold); y += 36
        draw.text((x, y), v, fill=(40,40,40), font=f_text); y += 46
    field("Membership ID:", membership_id or "—")
    field("Nome:", name or "—")
    field("EID:", eid or "—")
    field("Idade:", f"{age} anos" if age else "—")
    field("Faixa:", belt or "—")

    try:
        import qrcode
        qr = qrcode.make(membership_id or "").resize((200,200))
        canvas.paste(qr, (width-230, height-230))
    except Exception:
        pass

    buf = BytesIO(); canvas.save(buf, format="JPEG", quality=92); return buf.getvalue()

# ================================
# 6) NORMALIZAÇÃO / MÁSCARAS / OPÇÕES
# ================================
def squeeze_spaces(s:str)->str: return re.sub(r"\s+", " ", s or "").strip()
def title_capitalize(s:str)->str: return " ".join([w.capitalize() for w in squeeze_spaces(s).split(" ")])
def normalize_academy(s:str)->str: return title_capitalize(s)

def format_phone_live(raw: str) -> str:
    raw = raw or ""; digits = re.sub(r"\D", "", raw)
    if digits.startswith("9715"): digits = "05" + digits[-8:]
    if digits.startswith("5") and len(digits) <= 9: digits = "0" + digits
    digits = digits[:10]
    if len(digits) <= 3: return digits
    if len(digits) <= 6: return f"{digits[:3]}-{digits[3:]}"
    return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"

def clean_and_format_phone_final(raw: str) -> str:
    digits = re.sub(r"\D", "", raw or "")
    if digits.startswith("9715"): digits = "05" + digits[-8:]
    if digits.startswith("5") and len(digits) == 9: digits = "0" + digits
    digits = digits[:10]
    if len(digits) == 10 and digits.startswith("05"): return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    return raw

def eid_format_live(raw: str) -> str:
    digits = re.sub(r"\D", "", raw or "")[:15]
    if not digits.startswith("784"):
        digits = "784" + digits
        digits = digits[:15]
    parts = []
    if len(digits) >= 3: parts.append(digits[:3])
    if len(digits) >= 7: parts.append(digits[3:7])
    elif len(digits) > 3: parts.append(digits[3:])
    if len(digits) >= 14: parts.append(digits[7:14])
    elif len(digits) > 7: parts.append(digits[7:])
    if len(digits) == 15: parts.append(digits[14:])
    return "-".join(parts)

def eid_is_valid(masked: str) -> bool:
    return re.fullmatch(r"784-\d{4}-\d{7}-\d", masked or "") is not None

def date_mask_live(raw: str) -> str:
    d = re.sub(r"\D", "", raw or "")[:8]
    if len(d) <= 2: return d
    if len(d) <= 4: return f"{d[:2]}/{d[2:]}"
    return f"{d[:2]}/{d[2:4]}/{d[4:]}"

def parse_masked_date(masked: str):
    try:
        m = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", masked or "")
        if not m: return None
        day, month, year = map(int, m.groups())
        return dt.date(year, month, day)
    except Exception:
        return None

FAIXAS_GI = ["Branca","Cinza (Kids)","Amarela (Kids)","Laranja (Kids)","Verde (Kids)","Azul","Roxa","Marrom","Preta"]

def get_country_list():
    try:
        import pycountry
        countries = [c.name for c in pycountry.countries]
        extras = ["Hong Kong","Macau","Palestine","Kosovo"]
        return sorted(set(countries+extras))
    except Exception:
        return ["United Arab Emirates","Brazil","Portugal","United States","United Kingdom","Italy","Spain","France","Netherlands"]

def compute_age_year_based(year:int)->int:
    if not year: return 0
    return max(0, dt.date.today().year - year)

def age_division_by_year(age:int)->str:
    if age<=15: return "Kids"
    if 16<=age<=17: return "Juvenile"
    if 18<=age<=29: return "Adult"
    if 30<=age<=35: return "Master 1"
    if 36<=age<=40: return "Master 2"
    if 41<=age<=45: return "Master 3"
    return "Master 4+"

def weight_division(weight_kg: float) -> str:
    if not isinstance(weight_kg, (int, float)) or weight_kg <= 0:
        return "N/D"
    start = int(weight_kg // 5) * 5
    lo = start
    hi = start + 4.9
    if lo < 20: lo = 20; hi = 24.9
    if hi >= 125: return "125+ kg"
    return f"{lo:.0f}–{hi:.1f} kg"

# ================================
# 7) OCR / VERIFICAÇÃO EID
# ================================
EID_REGEX = re.compile(r"784-\d{4}-\d{7}-\d")
DATE_REGEXES = [
    re.compile(r"\b(\d{2})/(\d{2})/(\d{4})\b"),
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
]
NATIONALITY_KEYS = ["nationality", "country", "citizenship"]
NAME_KEYS = ["name", "cardholder", "card holder", "full name", "nome", "الاسم"]
KEYWORDS_EID = ["emirates id", "united arab emirates", "identity", "authority", "emirates", "الهوية", "الإمارات"]

def to_text_lines(s: str): return [l.strip() for l in (s or "").splitlines() if l.strip()]

def google_vision_ocr_extract(image_bytes: bytes) -> dict:
    out = {"text": "", "eid": "", "dob": "", "nationality": "", "name": "", "confidence": 0.0, "keywords_found": []}
    if not USE_VISION_OCR: return out
    try:
        image = vision.Image(content=image_bytes)
        resp = vision_client.document_text_detection(image=image)
        if resp.error.message: return out
        full_text = resp.full_text_annotation.text or ""
        out["text"] = full_text
        text_lower = full_text.lower()
        out["keywords_found"] = [kw for kw in KEYWORDS_EID if kw in text_lower]
        m = EID_REGEX.search(full_text.replace(" ", ""));  out["eid"] = m.group(0) if m else ""
        for rx in DATE_REGEXES:
            m = rx.search(full_text)
            if m:
                try:
                    if rx.pattern.startswith(r"\b(\d{2})/"):
                        d,mn,y = map(int, m.groups())
                    else:
                        y,mn,d = map(int, m.groups())
                    _ = dt.date(y,mn,d)
                    out["dob"] = f"{d:02d}/{mn:02d}/{y:04d}"
                    break
                except Exception:
                    pass
        lines = to_text_lines(full_text)
        for li in lines:
            low = li.lower()
            if any(k in low for k in NATIONALITY_KEYS):
                parts = re.split(r"[:\-]", li, 1)
                cand = parts[1].strip() if len(parts) > 1 else li
                out["nationality"] = squeeze_spaces(re.sub(r"(?i)nationality", "", cand)).strip(); break
        for li in lines:
            low = li.lower()
            if any(k in low for k in NAME_KEYS):
                parts = re.split(r"[:\-]", li, 1)
                cand = parts[1].strip() if len(parts) > 1 else li
                cand = re.sub(r"(?i)name|cardholder|card holder|full name|nome|الاسم", "", cand).strip()
                out["name"] = squeeze_spaces(cand); break
        score = 0.0
        if out["eid"]: score += 0.4
        if out["dob"]: score += 0.2
        if out["nationality"]: score += 0.2
        if out["name"]: score += 0.1
        if out["keywords_found"]: score += 0.1
        out["confidence"] = min(1.0, score)
        return out
    except Exception:
        return out

def local_heuristic_extract(image_bytes: bytes) -> dict:
    return {"text":"", "eid":"", "dob":"", "nationality":"", "name":"", "confidence":0.0, "keywords_found":[]}

def is_likely_eid_card(image_bytes: bytes, ocr_text: str) -> (bool, float, dict):
    details = {}
    try:
        img = Image.open(BytesIO(image_bytes))
        w, h = img.size; ratio = w / h if h else 0
        details["ratio"] = ratio
        ratio_ok = 1.427 <= ratio <= 1.745
    except Exception:
        ratio_ok = False
    ocr_low = (ocr_text or "").lower()
    eid_found = bool(EID_REGEX.search((ocr_text or "").replace(" ", "")))
    kw_found = any(kw in ocr_low for kw in KEYWORDS_EID)
    # <-- CUIDADO COM O PARÊNTESE: tudo numa linha para evitar SyntaxError
    score = (0.3 if ratio_ok else 0.0) + (0.5 if eid_found else 0.0) + (0.2 if kw_found else 0.0)
    return (score >= 0.6), score, details

# ---------- HELPERS p/ UPLOAD (jpg/png/heic/pdf) ----------
def is_pdf_file(uploaded):
    try:
        return str(uploaded.name).lower().endswith(".pdf")
    except Exception:
        return False

def is_heic_file(uploaded):
    try:
        return str(uploaded.name).lower().endswith((".heic", ".heif"))
    except Exception:
        return False

def convert_pdf_first_page_to_pil(uploaded):
    try:
        data = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
        from pdf2image import convert_from_bytes
        pages = convert_from_bytes(data, first_page=1, last_page=1, fmt="jpeg")
        return pages[0] if pages else None
    except Exception:
        return None

def load_any_image_for_ocr(uploaded_or_pil):
    if isinstance(uploaded_or_pil, Image.Image):
        return uploaded_or_pil
    f = uploaded_or_pil
    if f is None: return None
    if is_pdf_file(f):
        img = convert_pdf_first_page_to_pil(f)
        if img is not None: return img
    if is_heic_file(f):
        try:
            import pillow_heif
            heif = pillow_heif.read_heif(f.getvalue())
            img = Image.frombytes(heif.mode, heif.size, heif.data, "raw")
            return img.convert("RGB")
        except Exception:
            pass
    try:
        data = f.getvalue() if hasattr(f, "getvalue") else f.read()
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        return None

def save_uploaded_file_as_is(uploaded, reg_id: str, suffix: str) -> str:
    try:
        name = str(uploaded.name)
        ext = Path(name).suffix.lower() or ".bin"
        dest = UPLOAD_DIR / f"{reg_id}_{suffix}{ext}"
        data = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
        with open(dest, "wb") as f:
            f.write(data)
        return str(dest)
    except Exception:
        return ""

def extract_eid_fields_from_image(uploaded_or_pil, for_guardian=False):
    img_pil = load_any_image_for_ocr(uploaded_or_pil)
    if img_pil is None:
        return {"ok": False, "reason": "Não foi possível ler imagem do documento.", "data": {}}
    buf = BytesIO(); img_pil.save(buf, format="JPEG"); img_bytes = buf.getvalue()
    ocr = google_vision_ocr_extract(img_bytes) if USE_VISION_OCR else local_heuristic_extract(img_bytes)
    likely, score, geo = is_likely_eid_card(img_bytes, ocr.get("text",""))
    return {
        "ok": likely,
        "reason": f"confiança={score:.2f}, ratio={geo.get('ratio','?')}",
        "data": {
            "eid": eid_format_live(ocr.get("eid","")),
            "dob": ocr.get("dob",""),
            "nationality": ocr.get("nationality",""),
            "full_name": ocr.get("name",""),
        }
    }

# ================================
# 8) WEBCAM (streamlit-webrtc) – foco contínuo + captura
# ================================
if WEBRTC_AVAILABLE and RTCConfiguration is not None:
    RTC_CFG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
else:
    RTC_CFG = None

class SnapshotProcessor(VideoProcessorBase):
    def __init__(self):
        self.latest_frame = None
    def recv(self, frame):
        self.latest_frame = frame.to_ndarray(format="bgr24")
        return frame

def webrtc_capture_block(title: str, key_prefix: str, facing="user"):
    st.caption(title)
    if not WEBRTC_AVAILABLE or webrtc_streamer is None or RTC_CFG is None:
        return None, st.camera_input(title + " (fallback)")
    constraints = {
        "video": {
            "facingMode": facing,
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
            "frameRate": {"ideal": 30},
            "advanced": [{"focusMode": "continuous"}]
        },
        "audio": False
    }
    ctx = webrtc_streamer(
        key=f"webrtc_{key_prefix}",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTC_CFG,
        media_stream_constraints=constraints,
        video_processor_factory=SnapshotProcessor,
    )
    snap = None
    if ctx and getattr(ctx, "video_processor", None):
        if st.button(f"Capturar foto – {title}", key=f"btn_{key_prefix}"):
            frame = ctx.video_processor.latest_frame
            if frame is not None:
                if FACE_DETECT_AVAILABLE:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    rgb = frame[..., ::-1]
                snap = Image.fromarray(rgb)
    return snap, None  # (PIL, fallback_uploaded)

# ================================
# 9) ESTADO / NAVEGAÇÃO
# ================================
if "screen" not in st.session_state: st.session_state["screen"]="welcome"
if "accepted_terms" not in st.session_state: st.session_state["accepted_terms"]=False
if "active_event_id" not in st.session_state: st.session_state["active_event_id"]=None
if "active_event_name" not in st.session_state: st.session_state["active_event_name"]=""
if "region" not in st.session_state: st.session_state["region"]=EVENT_DEFAULT_REGION
if "errors" not in st.session_state: st.session_state["errors"]=set()
if "admin_ok" not in st.session_state: st.session_state["admin_ok"]=False

def whatsapp_button(phone: str):
    if not phone or not phone.strip(): return
    phone_clean = phone.replace("+","").replace(" ","").replace("-","")
    msg = f"Olá! Tenho uma dúvida sobre o evento {st.session_state.get('active_event_name') or 'ADSS'}."
    url = f"https://wa.me/{phone_clean}?text=" + urllib.parse.quote(msg)
    st.markdown(
        f"""<a href="{url}" target="_blank"
           style="position:fixed;right:20px;bottom:20px;background:#25D366;color:#fff;
           padding:12px 16px;border-radius:999px;text-decoration:none;font-weight:700;
           box-shadow:0 4px 16px rgba(0,0,0,.2);z-index:9999;">WhatsApp da Organização</a>""",
        unsafe_allow_html=True
    )

# ================================
# 10) TELAS
# ================================
def screen_welcome():
    title = st.session_state.get("active_event_name") or "ADSS Jiu-Jitsu Competition"
    st.markdown(f"<h1 style='text-align:center;margin:0'>{title}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='header-sub'>{st.session_state['region']}</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Bem-vindo(a)!")
    st.markdown("""
**Guideline da competição (resumo):**
- Chegar com antecedência mínima de 45 minutos.
- Documento de identificação oficial obrigatório.
- Uso de kimono/manguito conforme faixa e regulamento.
- Respeitar cronograma e chamadas de área.
- O organizador pode ajustar chaves/categorias em caso de WO.
- Ao se inscrever, você declara concordar com as regras do evento.
""")
    agree = st.checkbox("Li e **concordo** com os termos e condições", key="agree_terms")
    st.markdown("---")
    # Se houver eventos ativos, forçar seleção
    active_events = fetch_events(status="Ativo")
    if active_events.empty:
        st.info("Não há eventos ativos no momento. O administrador pode cadastrar/ativar um evento na aba Events.")
    else:
        ev_names = [f"{r['name']} — {r['date'][:10]} — {r['location']}" for _, r in active_events.iterrows()]
        idx = st.selectbox("Selecione o evento ativo para continuar", options=list(range(len(ev_names))),
                           format_func=lambda i: ev_names[i], index=0 if st.session_state.get("active_event_id") is None else
                           max(0, active_events.index[active_events['id']==st.session_state["active_event_id"]].tolist()[0] if (st.session_state["active_event_id"] in set(active_events['id'])) else 0))
        chosen = active_events.iloc[idx].to_dict()
        st.markdown(f"<div class='help'>Descrição: {chosen.get('description','')}</div>", unsafe_allow_html=True)

    cols = st.columns(2)
    if cols[0].button("Prosseguir"):
        if not agree:
            st.error("Você precisa concordar com os termos e condições para continuar.")
            return
        if not active_events.empty:
            st.session_state["active_event_id"] = chosen["id"]
            st.session_state["active_event_name"] = chosen["name"]
        st.session_state["accepted_terms"]=True
        st.session_state["screen"]="menu"

def screen_menu():
    name = st.session_state.get("active_event_name") or "ADSS Jiu-Jitsu Competition"
    st.markdown(f"<h1 style='text-align:center;margin:0'>{name}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='header-sub'>{st.session_state['region']}</p>", unsafe_allow_html=True)
    st.caption("Menu principal — todas as ações referem-se ao evento selecionado acima.")
    if st.button("Novo registro\nPreencher formulário de inscrição", key="menu_new"):
        st.session_state["screen"]="new_registration"; return
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    if st.button("Alterar registro\nFace ID (selfie) ou número da inscrição", key="menu_update"):
        st.session_state["screen"]="update_registration"; return
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    if st.button("Lista de inscritos (pública)\nVisualizar todos os inscritos", key="menu_public"):
        st.session_state["screen"]="public_list"; return
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    if st.button("Gerenciamento da competição (admin)\nAcessar painel do organizador", key="menu_admin"):
        st.session_state["screen"]="admin"; return
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    if st.button("Events (admin)\nCadastrar e ativar eventos", key="menu_events"):
        st.session_state["screen"]="events_admin"; return

def screen_public_list():
    st.title("Lista de Inscritos (pública)")
    df = fetch_all(st.session_state.get("active_event_id"))
    if df.empty:
        st.info("Ainda não há inscrições neste evento.")
        st.button("Voltar ao menu principal", on_click=lambda: st.session_state.update(screen="menu"))
        return

    df = df.copy()
    def birth_year(d):
        try:
            return dt.date.fromisoformat(d).year
        except Exception:
            return None
    df["birth_year"] = df["dob"].apply(birth_year)
    df["age_years"] = df["age_years"].fillna(0).astype(int)
    df["weight_div"] = df["weight_kg"].apply(weight_division)
    df["gender"] = df["gender"].fillna("N/D")
    df["age_division"] = df["age_division"].fillna("N/D")
    df["belt"] = df["belt"].fillna("N/D")

    grouped = df.groupby(["gender","age_division","belt","weight_div"], dropna=False)

    for (gender, age_div, belt, wdiv), gdf in grouped:
        st.markdown(f"<div class='group-title'>{gender} / {age_div} / {belt} / {wdiv} — {len(gdf)} inscrito(s)</div>", unsafe_allow_html=True)
        for _, row in gdf.iterrows():
            name = f"{row.get('first_name','')} {row.get('last_name','')}".strip()
            status = (row.get("approval_status") or "Pending").strip().lower()
            badge = f"<span class='badge-pending'>Pendente</span>" if status=="pending" else "<span class='badge-approved'>Aprovado</span>"
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            if row.get("profile_photo_path") and Path(row["profile_photo_path"]).exists():
                st.image(row["profile_photo_path"], width=90)
            else:
                st.image(Image.new("RGB",(90,90),(230,230,230)), width=90)
            meta = []
            by = row.get("birth_year"); ag = row.get("age_years")
            meta.append(f"<b>{name or '(Sem nome)'}</b> {badge}")
            if by: meta.append(f"Nasc.: {by}  •  Idade: {ag} anos")
            if row.get("academy"): meta.append(f"Academia: {row['academy']}")
            meta.append(f"Divisão: {gender} / {age_div} / {belt} / {row.get('weight_div','N/D')}")
            st.markdown(f"<div class='meta'>{'<br>'.join(meta)}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.button("Voltar ao menu principal", on_click=lambda: st.session_state.update(screen="menu"))

def stats_view(embed: bool=False):
    df = fetch_all(st.session_state.get("active_event_id"))
    if df.empty:
        st.info("Ainda não há inscrições neste evento."); return
    totals = (df.groupby("category", dropna=False).size().reset_index(name="inscritos")
                .sort_values("inscritos", ascending=False).reset_index(drop=True))
    if not embed:
        st.title("Estatísticas – Inscritos por Categoria")
    st.subheader("Total por categoria")
    st.dataframe(totals, use_container_width=True)
    st.bar_chart(totals.set_index("category"))

def admin_view():
    st.subheader("Painel do Organizador")
    df = fetch_all(st.session_state.get("active_event_id"))
    st.write(f"Total de inscrições neste evento: {len(df)}")
    if df.empty:
        st.info("Ainda não há inscrições."); return
    for _, row in df.iterrows():
        name = f"{row.get('first_name','')} {row.get('last_name','')}".strip()
        status = (row.get("approval_status") or "Pending").strip()
        badge = "<span class='badge-pending'>Pendente de aprovação</span>" if status.lower()=="pending" else "<span class='badge-approved'>Aprovado</span>"
        header = (name or row.get('id','(sem id)')) + " " + badge
        with st.expander(header, expanded=False):
            meta = {
                "ID": row.get("id",""),
                "Evento": row.get("event_name",""),
                "Nome": name,
                "E-mail": row.get("email",""),
                "Telefone": row.get("phone",""),
                "EID": row.get("eid",""),
                "Nacionalidade": row.get("nationality",""),
                "Gênero": row.get("gender",""),
                "Data de nascimento": row.get("dob",""),
                "Idade (ano-base)": row.get("age_years",""),
                "Divisão etária": row.get("age_division",""),
                "Academia": row.get("academy",""),
                "Professor": row.get("coach",""),
                "Telefone do professor": row.get("coach_phone",""),
                "Peso atual (kg)": row.get("weight_kg",""),
                "Faixa": row.get("belt",""),
                "Categoria": row.get("category",""),
                "Status": status,
                "Criado em (UTC)": row.get("created_at",""),
                "Responsável (se menor)": row.get("guardian_name","")
            }
            st.write(meta)
            if row.get("profile_photo_path") and Path(row["profile_photo_path"]).exists():
                st.image(row["profile_photo_path"], caption="Foto de Perfil", width=160)
            doc_path = row.get("id_doc_photo_path")
            if doc_path and Path(doc_path).exists():
                if doc_path.lower().endswith(".pdf"):
                    with open(doc_path, "rb") as f:
                        st.download_button("Baixar documento (PDF)", f, file_name=Path(doc_path).name, mime="application/pdf", key=f"dlpdf_{row['id']}")
                else:
                    st.image(doc_path, caption="Documento do atleta", width=220)
            if row.get("guardian_eid_photo_path") and Path(row["guardian_eid_photo_path"]).exists():
                gpath = row["guardian_eid_photo_path"]
                if gpath.lower().endswith(".pdf"):
                    with open(gpath, "rb") as f:
                        st.download_button("Baixar EID do responsável (PDF)", f, file_name=Path(gpath).name, mime="application/pdf", key=f"dlpdfg_{row['id']}")
                else:
                    st.image(gpath, caption="EID do responsável", width=220)

            jpg = generate_registration_jpg(row, row.get("event_name","ADSS"), st.session_state["region"])
            st.download_button("Baixar inscrição (JPG)", data=jpg,
                               file_name=f"inscricao_{row['id']}.jpg", mime="image/jpeg",
                               key=f"dl_admin_{row['id']}")
            try:
                membership = generate_membership_card(row.get("profile_photo_path",""), row.get("id",""),
                                                      row.get("eid",""), row.get("age_years",0),
                                                      name, row.get("belt",""))
                st.download_button("Baixar carteirinha (JPG)", data=membership,
                                   file_name=f"membership_{row['id']}.jpg", mime="image/jpeg",
                                   key=f"dl_member_{row['id']}")
            except Exception as e:
                st.warning(f"Não foi possível gerar a carteirinha: {e}")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Baixar CSV", data=csv, file_name="inscricoes.csv", mime="text/csv")
    st.markdown("---"); stats_view(embed=True)
    st.markdown("---"); st.button("Voltar ao menu principal", on_click=lambda: st.session_state.update(screen="menu"))

def screen_admin_gate():
    st.title("Gerenciamento da Competição (Admin)")
    admin_pw = st.text_input("Senha admin", type="password", placeholder="••••••")
    if st.button("Entrar"):
        st.session_state["admin_ok"] = (admin_pw == ADMIN_PASSWORD)
        if not st.session_state["admin_ok"]:
            st.error("Senha incorreta.")
    if st.session_state.get("admin_ok"): admin_view()
    st.markdown("---")
    st.button("Voltar ao menu principal", on_click=lambda: st.session_state.update(screen="menu"))

def render_edit_form(best_row):
    st.markdown("---"); st.subheader("Editar dados")
    with st.form(f"edit_{best_row['id']}", clear_on_submit=False):
        colA, colB = st.columns(2)
        with colA:
            first = st.text_input("Nome", value=best_row.get("first_name",""))
            last  = st.text_input("Sobrenome", value=best_row.get("last_name",""))
            email = st.text_input("E-mail", value=best_row.get("email",""))
            phone = st.text_input("Telefone (05_-___-____)", value=best_row.get("phone",""))
            eid   = st.text_input("EID (784-####-#######-#)", value=best_row.get("eid",""))
        with colB:
            nationality= st.text_input("Nacionalidade", value=best_row.get("nationality",""))
            gender     = st.selectbox("Gênero", ["Masculino","Feminino"], index=0 if best_row.get("gender","Masculino")=="Masculino" else 1)
            belt_opts  = FAIXAS_GI
            belt       = st.selectbox("Faixa", belt_opts, index=belt_opts.index(best_row.get("belt", belt_opts[0])) if best_row.get("belt","") in belt_opts else 0)
            weight     = st.number_input("Peso atual (kg)", value=float(best_row.get("weight_kg") or 0), min_value=0.0, step=0.1)
            academy    = st.text_input("Academia", value=best_row.get("academy",""))
            coach      = st.text_input("Professor/Coach", value=best_row.get("coach",""))
            coach_phone= st.text_input("Telefone do Professor (05_-___-____)", value=best_row.get("coach_phone",""))
        st.markdown("Nova foto de perfil (opcional)")
        snap, fallback = webrtc_capture_block("Nova foto de perfil (editar)", f"edit_{best_row['id']}_profile")
        new_profile = snap or fallback
        submitted_update = st.form_submit_button("Salvar alterações")

    if submitted_update:
        first = title_capitalize(first); last = title_capitalize(last)
        academy = normalize_academy(academy)
        phone = clean_and_format_phone_final(phone)
        coach_phone = clean_and_format_phone_final(coach_phone)
        eid_masked = eid_format_live(eid)
        upd = {
            "first_name": first, "last_name": last, "full_name_en": f"{first} {last}",
            "email": (email or "").strip(), "phone": phone, "eid": eid_masked,
            "nationality": nationality, "gender": gender, "academy": academy, "coach": title_capitalize(coach),
            "coach_phone": coach_phone, "belt": belt, "weight_kg": float(weight),
        }
        if new_profile is not None:
            reg_id = best_row["id"]
            new_path = save_image(new_profile, reg_id, "profile")
            upd["profile_photo_path"] = new_path
            crop = crop_face_from_uploaded(new_profile)
            upd["face_phash"] = compute_phash(crop)
        try:
            update_registration(best_row["id"], upd)
            st.success("Registro atualizado com sucesso.")
        except Exception as e:
            st.error(f"Erro ao atualizar: {e}")
    st.markdown("---"); st.button("Voltar ao menu principal", on_click=lambda: st.session_state.update(screen="menu"))

def screen_update_registration():
    st.title("Alterar Inscrição – Identificação")
    df = fetch_all(st.session_state.get("active_event_id"))
    if df.empty:
        st.info("Não há inscrições para alterar neste evento.")
        st.button("Voltar ao menu principal", on_click=lambda: st.session_state.update(screen="menu")); return
    tab1, tab2 = st.tabs(["Face ID (selfie)", "ID da inscrição"])
    with tab1:
        st.write("Tire uma selfie recente do atleta.")
        snap, fallback = webrtc_capture_block("Selfie do atleta", "selfie", facing="user")
        qfile = snap or fallback
        if qfile:
            q_hash = compute_phash(crop_face_from_uploaded(qfile))
            if not q_hash:
                st.error("Não foi possível processar a imagem. Tente outra selfie com o rosto visível.")
            else:
                dfc = df[df["face_phash"].notna() & (df["face_phash"]!="")].copy()
                if dfc.empty:
                    st.warning("Nenhuma inscrição possui pHash salvo (cadastros antigos).")
                else:
                    dfc["distance"] = dfc["face_phash"].apply(lambda h: phash_distance(h, q_hash))
                    dfc = dfc.sort_values("distance").reset_index(drop=True)
                    best = dfc.iloc[0]
                    st.write({"Provável inscrição": best["id"], "Nome": f"{best.get('first_name','')} {best.get('last_name','')}", "Distância": int(best["distance"])})
                    threshold = 8
                    confirm = st.checkbox("Confirmo que este é o atleta correto")
                    if confirm or int(best["distance"])<=threshold:
                        render_edit_form(best)
    with tab2:
        regid = st.text_input("Digite o número da inscrição (ID)")
        if st.button("Localizar"):
            row = df[df["id"]==regid]
            if row.empty:
                st.error("ID não encontrado (neste evento).")
            else:
                render_edit_form(row.iloc[0])
    st.markdown("---"); st.button("Voltar ao menu principal", on_click=lambda: st.session_state.update(screen="menu"))

# =============== NOVA ABA: EVENTS (ADMIN) ===============
def screen_events_admin():
    st.title("Events (Admin) – Cadastro e Gestão de Eventos")
    if not st.session_state.get("admin_ok"):
        st.info("Informe a senha de admin para gerenciar eventos.")
        admin_pw = st.text_input("Senha admin", type="password", placeholder="••••••", key="events_pw")
        if st.button("Entrar (Events)"):
            st.session_state["admin_ok"] = (admin_pw == ADMIN_PASSWORD)
            if not st.session_state["admin_ok"]:
                st.error("Senha incorreta.")
        st.markdown("---")
        st.button("Voltar ao menu principal", on_click=lambda: st.session_state.update(screen="menu"))
        return

    st.subheader("Criar novo evento")
    with st.form("new_event_form", clear_on_submit=True):
        name = st.text_input("Nome do evento*", placeholder="Ex.: ADSS Jiu-Jitsu Summer Open")
        date_str = st.text_input("Data (yyyy-mm-dd)*", placeholder="2025-09-15")
        location = st.text_input("Local*", placeholder="Al Dhafra – Abu Dhabi")
        manager = st.text_input("Responsável*", placeholder="Organizador / Contato")
        divisions = st.text_area("Divisões ativas", placeholder="Ex.: Gi Kids / Adult / Master; No-Gi Adult ...")
        description = st.text_area("Descrição", placeholder="Informações gerais, links, observações…")
        status = st.selectbox("Status*", ["Ativo","Inativo"])
        created = st.form_submit_button("Criar evento")
    if created:
        if not name or not date_str or not location or not manager or not status:
            st.error("Preencha os campos obrigatórios (*).")
        else:
            try:
                # valida data simples
                _ = dt.date.fromisoformat(date_str)
                ev = {
                    "id": str(uuid.uuid4())[:8].upper(),
                    "created_at": dt.datetime.utcnow().isoformat(),
                    "name": name.strip(),
                    "date": date_str.strip(),
                    "location": location.strip(),
                    "manager": manager.strip(),
                    "divisions": divisions.strip(),
                    "description": description.strip(),
                    "status": status
                }
                insert_event(ev)
                st.success("Evento criado com sucesso.")
            except Exception as e:
                st.error(f"Erro ao criar evento: {e}")

    st.subheader("Eventos existentes")
    ev_df = fetch_events()
    if ev_df.empty:
        st.info("Nenhum evento cadastrado.")
    else:
        for _, r in ev_df.iterrows():
            with st.expander(f"{r['name']} — {r['date'][:10]} — {r['location']}  [{r['status']}]"):
                st.write({
                    "ID": r["id"],
                    "Nome": r["name"],
                    "Data": r["date"],
                    "Local": r["location"],
                    "Responsável": r["manager"],
                    "Divisões": r.get("divisions",""),
                    "Descrição": r.get("description",""),
                    "Status": r["status"]
                })
                with st.form(f"edit_{r['id']}"):
                    ename = st.text_input("Nome", value=r["name"])
                    edate = st.text_input("Data (yyyy-mm-dd)", value=r["date"])
                    eloc  = st.text_input("Local", value=r["location"])
                    emag  = st.text_input("Responsável", value=r["manager"])
                    ediv  = st.text_area("Divisões ativas", value=r.get("divisions",""))
                    edesc = st.text_area("Descrição", value=r.get("description",""))
                    est   = st.selectbox("Status", ["Ativo","Inativo"], index=0 if r["status"]=="Ativo" else 1)
                    save  = st.form_submit_button("Salvar alterações")
                if save:
                    try:
                        _ = dt.date.fromisoformat(edate)
                        update_event(r["id"], {
                            "name": ename.strip(),
                            "date": edate.strip(),
                            "location": eloc.strip(),
                            "manager": emag.strip(),
                            "divisions": ediv.strip(),
                            "description": edesc.strip(),
                            "status": est
                        })
                        st.success("Evento atualizado.")
                    except Exception as e:
                        st.error(f"Erro ao atualizar: {e}")

    st.markdown("---")
    st.button("Voltar ao menu principal", on_click=lambda: st.session_state.update(screen="menu"))

# =============== NOVO REGISTRO (agora vinculado ao evento ativo) ===============
def screen_new_registration():
    # Guard: precisa de evento ativo
    if not st.session_state.get("active_event_id"):
        st.warning("Selecione um evento ativo na tela inicial antes de registrar.")
        st.button("Voltar ao início", on_click=lambda: st.session_state.update(screen="welcome"))
        return

    # Máscaras live
    for k in ["phone","coach_phone","guardian_phone"]:
        if k in st.session_state: st.session_state[k] = format_phone_live(st.session_state[k])
    for k in ["eid","guardian_eid"]:
        if k in st.session_state: st.session_state[k] = eid_format_live(st.session_state[k])
    if "dob" in st.session_state: st.session_state["dob"] = date_mask_live(st.session_state["dob"])

    # Buffers de OCR aplicados antes dos widgets
    if st.session_state.get("ocr_apply_atleta"):
        data = st.session_state.pop("ocr_apply_atleta")
        if data.get("eid"): st.session_state["eid"] = eid_format_live(data["eid"])
        if data.get("dob"): st.session_state["dob"] = date_mask_live(data["dob"])
        if data.get("nationality"): st.session_state["nationality"] = data["nationality"]
        if data.get("full_name"):
            tok = data["full_name"].split()
            if tok:
                st.session_state["first_name"] = title_capitalize(tok[0])
                st.session_state["last_name"]  = title_capitalize(" ".join(tok[1:])) if len(tok)>1 else st.session_state.get("last_name","")
    if st.session_state.get("ocr_apply_guardian"):
        data = st.session_state.pop("ocr_apply_guardian")
        if data.get("eid"): st.session_state["guardian_eid"] = eid_format_live(data["eid"])
        if data.get("full_name"): st.session_state["guardian_name"] = title_capitalize(data["full_name"])

    st.title(f"Formulário de Inscrição — {st.session_state.get('active_event_name')}")
    st.caption("Preencha seus dados. Campos com * são obrigatórios.")

    # PESSOAIS
    st.subheader("Dados pessoais")
    colA, colB = st.columns(2)
    with colA:
        first_name = st.text_input("Nome*", key="first_name", placeholder="Ex.: João")
        last_name  = st.text_input("Sobrenome*", key="last_name", placeholder="Ex.: Silva")
        gender     = st.selectbox("Gênero*", ["Masculino","Feminino"], key="gender")
        nationality= st.selectbox("Nacionalidade*", get_country_list(), key="nationality")
        eid_mask   = st.text_input("EID (784-####-#######-#)*", key="eid", placeholder="784-1234-1234567-1")
    with colB:
        email      = st.text_input("E-mail*", key="email", placeholder="email@exemplo.com")
        phone_in   = st.text_input("Telefone/WhatsApp (05_-___-____)*", key="phone", placeholder="052-123-4567")
        dob_in     = st.text_input("Data de nascimento (dd/mm/aaaa)*", key="dob", placeholder="dd/mm/aaaa")
        dob_date = parse_masked_date(st.session_state.get("dob",""))
        age_years = compute_age_year_based(dob_date.year) if dob_date else 0
        st.text_input("Idade (auto)", value=str(age_years if age_years else ""), disabled=True)

    # ACADEMIA
    st.subheader("Academia")
    academies = fetch_distinct_academies(st.session_state.get("active_event_id"))
    options = ["Selecione...", "Minha academia não está na lista"] + academies
    academy_choice = st.selectbox("Academia*", options, key="academy_choice", index=st.session_state.get("academy_choice_idx", 0))
    academy_other = ""
    if academy_choice == "Minha academia não está na lista":
        academy_other = st.text_input("Nome da academia*", key="academy_other", placeholder="Nome da academia")
    coach      = st.text_input("Professor/Coach", key="coach", placeholder="Nome do coach")
    coach_phone= st.text_input("Telefone do Professor (05_-___-____)", key="coach_phone", placeholder="052-123-4567")

    # --- DOCUMENTOS ---
    st.subheader("Documentos")
    snap_prof, fb_prof = webrtc_capture_block("Foto de Perfil (rosto visível)*", "profile_selfie", facing="user")
    profile_img = snap_prof or fb_prof

    doc_opt = st.radio("Como fornecer o documento do atleta?", ["Tirar foto agora", "Enviar arquivo (jpg/png/heic/pdf)"],
                       index=0, horizontal=True)
    id_doc_source = None
    if doc_opt == "Tirar foto agora":
        snap_id, fb_id = webrtc_capture_block("EID do atleta (frente)*", "athlete_eid", facing="environment")
        id_doc_source = snap_id or fb_id
    else:
        id_doc_upload = st.file_uploader("Enviar arquivo do documento*", type=["jpg","jpeg","png","heic","pdf"], key="id_doc_upload")
        id_doc_source = id_doc_upload

    colx, _ = st.columns([1,4])
    with colx:
        if st.button("Ler dados da EID do atleta"):
            res = extract_eid_fields_from_image(id_doc_source)
            if not res["ok"]:
                st.warning(f"Foto/arquivo pode não ser uma EID ({res['reason']}). Mesmo assim tentando aproveitar dados extraídos.")
            st.session_state["ocr_apply_atleta"] = res.get("data", {})
            st.rerun()

    # COMPETIÇÃO
    st.subheader("Informações de Competição")
    belt        = st.selectbox("Faixa*", FAIXAS_GI, key="belt")
    weight_kg   = st.number_input("Peso atual do atleta (kg)*", min_value=0.0, step=0.1, key="weight_kg")

    # RESPONSÁVEL (menor de 18)
    if age_years and age_years < 18:
        st.subheader("Responsável legal (obrigatório para menor de 18)")
        guardian_name  = st.text_input("Nome completo do responsável*", key="guardian_name")
        guardian_eid   = st.text_input("EID do responsável (784-####-#######-#)*", key="guardian_eid")
        guardian_phone = st.text_input("Telefone do responsável (05_-___-____)*", key="guardian_phone")
        snap_geid, fb_geid = webrtc_capture_block("Foto da EID do responsável (frente)*", "guardian_eid", facing="environment")
        guardian_eid_photo = snap_geid or fb_geid
        colg, _ = st.columns([1,4])
        with colg:
            if st.button("Ler dados da EID do responsável"):
                resg = extract_eid_fields_from_image(guardian_eid_photo, for_guardian=True)
                if not resg["ok"]:
                    st.warning(f"EID do responsável pode não ser válida ({resg['reason']}).")
                st.session_state["ocr_apply_guardian"] = resg.get("data", {})
                st.rerun()
    else:
        guardian_name = guardian_eid = guardian_phone = ""
        guardian_eid_photo = None

    # PRÉVIA DA DIVISÃO
    if all([gender, belt, weight_kg, age_years]):
        age_div = age_division_by_year(age_years)
        st.markdown("<div class='preview-box'>", unsafe_allow_html=True)
        st.markdown(f"**Divisão perfeita:** {gender} / {age_div} / {belt} / {weight_kg:.1f} kg")
        st.caption("Revise antes de enviar.")
        st.markdown("</div>", unsafe_allow_html=True)

    # CONSENTIMENTO + ENVIAR
    st.subheader("Termo de Consentimento")
    consent = st.checkbox("Eu li e concordo com o termo de consentimento*", key="consent")

    submitted = st.button("Enviar inscrição")
    if submitted:
        errors = set()
        must = ["first_name","last_name","email","phone","gender","nationality","belt","weight_kg","consent","dob","academy_choice","eid"]
        for k in must:
            if not st.session_state.get(k): errors.add(k)
        if profile_img is None: errors.add("profile_img")
        if id_doc_source is None: errors.add("id_doc_img")
        if st.session_state.get("academy_choice") == "Minha academia não está na lista" and not st.session_state.get("academy_other"):
            errors.add("academy_other")
        if st.session_state.get("eid") and not eid_is_valid(st.session_state["eid"]):
            errors.add("eid"); st.error("EID do atleta inválido. Use o formato 784-####-#######-#.")
        dob_date = parse_masked_date(st.session_state.get("dob",""))
        if dob_date is None:
            errors.add("dob"); st.error("Data de nascimento inválida. Use dd/mm/aaaa.")
        if "profile_img" not in errors and profile_img is not None:
            faces_profile = count_faces_in_image(profile_img)
            if faces_profile == -1:
                st.warning("Validação automática de rosto indisponível no ambiente atual.")
            else:
                if faces_profile == 0:
                    errors.add("profile_img"); st.error("A foto de perfil não parece conter um rosto.")
                elif faces_profile > 1:
                    errors.add("profile_img"); st.error("Detectamos mais de uma pessoa na foto de perfil.")
        if dob_date:
            age_years = compute_age_year_based(dob_date.year)
        if age_years and age_years < 18:
            if not st.session_state.get("guardian_name"): errors.add("guardian_name")
            if not st.session_state.get("guardian_eid") or not eid_is_valid(st.session_state.get("guardian_eid","")):
                errors.add("guardian_eid"); st.error("EID do responsável inválido.")
            if not st.session_state.get("guardian_phone"): errors.add("guardian_phone")
            if guardian_eid_photo is None: errors.add("guardian_eid_photo")

        st.session_state["errors"] = errors
        if errors:
            st.error("Há campos obrigatórios faltando/invalidos. Corrija e reenvie.")
            return

        # Normalizações
        first_name = title_capitalize(st.session_state["first_name"])
        last_name  = title_capitalize(st.session_state["last_name"])
        full_name_en = f"{first_name} {last_name}"
        phone = clean_and_format_phone_final(st.session_state["phone"])
        coach_phone = clean_and_format_phone_final(st.session_state.get("coach_phone","")) if st.session_state.get("coach_phone") else ""
        eid_masked = eid_format_live(st.session_state["eid"])
        academy = normalize_academy(st.session_state["academy_other"] if st.session_state["academy_choice"]=="Minha academia não está na lista" else st.session_state["academy_choice"])
        dob_iso = dob_date.isoformat()
        age_div = age_division_by_year(age_years)
        category = age_div

        # salvar imagens e documento
        reg_id = str(uuid.uuid4())[:8].upper()
        profile_photo_path = save_image(profile_img, reg_id, "profile")
        if isinstance(id_doc_source, Image.Image):
            id_doc_photo_path = save_image(id_doc_source, reg_id, "id_doc")
        else:
            id_doc_photo_path = save_uploaded_file_as_is(id_doc_source, reg_id, "id_doc")
        guardian_eid_photo_path = save_image(guardian_eid_photo, reg_id, "guardian_eid") if guardian_eid_photo else ""
        face_crop_img = crop_face_from_uploaded(profile_img)
        face_phash = compute_phash(face_crop_img)

        row = {
            "id": reg_id, "created_at": dt.datetime.utcnow().isoformat(),
            "event_id": st.session_state.get("active_event_id"),    # vínculo
            "event_name": st.session_state.get("active_event_name"),
            "first_name": first_name, "last_name": last_name, "full_name_en": full_name_en,
            "email": st.session_state["email"].strip(), "phone": phone, "eid": eid_masked,
            "nationality": st.session_state["nationality"], "gender": st.session_state["gender"],
            "dob": dob_iso, "age_years": age_years, "age_division": age_div,
            "academy": academy, "coach": title_capitalize(st.session_state.get("coach","")),
            "coach_phone": coach_phone, "region": st.session_state["region"],
            "belt": st.session_state["belt"], "weight_kg": float(st.session_state["weight_kg"]),
            "category": category, "consent": 1,
            "profile_photo_path": profile_photo_path, "id_doc_photo_path": id_doc_photo_path,
            "approval_status": "Pending", "face_phash": face_phash,
            "guardian_name": st.session_state.get("guardian_name",""),
            "guardian_eid": eid_format_live(st.session_state.get("guardian_eid","")) if st.session_state.get("guardian_eid") else "",
            "guardian_phone": clean_and_format_phone_final(st.session_state.get("guardian_phone","")) if st.session_state.get("guardian_phone") else "",
            "guardian_eid_photo_path": guardian_eid_photo_path
        }

        try:
            insert_registration(row)
            st.success(f"Inscrição enviada com sucesso. ID (Membership): {reg_id}")
            st.info("Status da sua inscrição: Pendente de aprovação.")
            st.info(f"Atletas já inscritos na categoria '{category}': {count_by_category(category, st.session_state.get('active_event_id'))}")

            jpg_bytes = generate_registration_jpg(row, row.get("event_name","ADSS"), st.session_state["region"])
            st.download_button("Baixar inscrição (JPG)", data=jpg_bytes, file_name=f"inscricao_{row['id']}.jpg", mime="image/jpeg")

            try:
                member_card = generate_membership_card(profile_photo_path, reg_id, eid_masked, age_years,
                                                       f"{first_name} {last_name}", st.session_state["belt"])
                st.download_button("Baixar carteirinha (JPG)", data=member_card, file_name=f"membership_{reg_id}.jpg", mime="image/jpeg")
                st.image(member_card, caption="Pré-visualização da carteirinha", use_column_width=True)
            except Exception as e:
                st.warning(f"Não foi possível gerar a carteirinha: {e}")
        except Exception as e:
            st.error(f"Erro ao salvar inscrição: {e}")

    st.markdown("---"); st.button("Voltar ao menu principal", on_click=lambda: st.session_state.update(screen="menu"))

# ================================
# 11) ROTEAMENTO + WHATSAPP
# ================================
whatsapp_button(WHATSAPP_PHONE)
if st.session_state["screen"] == "welcome":
    screen_welcome()
elif not st.session_state.get("accepted_terms", False):
    screen_welcome()
elif st.session_state["screen"] == "menu":
    screen_menu()
elif st.session_state["screen"] == "new_registration":
    screen_new_registration()
elif st.session_state["screen"] == "update_registration":
    screen_update_registration()
elif st.session_state["screen"] == "public_list":
    screen_public_list()
elif st.session_state["screen"] == "admin":
    screen_admin_gate()
elif st.session_state["screen"] == "events_admin":
    screen_events_admin()
else:
    st.session_state["screen"] = "menu"; screen_menu()
