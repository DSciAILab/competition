# ================================
# app.py – ADSS Jiu-Jitsu Competition
# WebRTC autofocus + OCR EID opcional + Documento por foto ou upload (jpg/png/heic/pdf)
# Lista pública agrupada + carteirinha JPG + Face ID por pHash + Admin
# + [EVENTS] Gerenciamento de eventos + seleção e escopo por evento
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
.chip {display:inline-block;padding:6px 10px;border:1px solid #2b2b2b;border-radius:999px;background:#0f1116;margin:4px 6px 0 0;}
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

EVENT_DEFAULT_NAME = "ADSS Jiu-Jitsu Competition"
EVENT_DEFAULT_REGION = "Al Dhafra – Abu Dhabi"

# ================================
# 3) BANCO DE DADOS (SQLite)
# ================================
engine = create_engine(f"sqlite:///{DB_PATH}", future=True)

def init_db():
    with engine.begin() as conn:
        # REGISTRATIONS
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS registrations (
            id TEXT PRIMARY KEY,
            created_at TEXT,
            event_id TEXT,                -- [EVENTS] vínculo duro com evento
            event_name TEXT,
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
        # [EVENTS] Tabela de eventos
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            name TEXT,
            date TEXT,
            location TEXT,
            manager TEXT,
            divisions_active TEXT,   -- csv: Kids,Juvenile,Adult,...
            description TEXT,
            status TEXT,             -- draft | active | archived
            created_at TEXT
        );
        """))
        # Alters antigos (idempotentes)
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
            "ALTER TABLE registrations ADD COLUMN event_id TEXT"
        ]:
            try: conn.execute(text(alter))
            except Exception: pass
init_db()

# --------- REGISTROS ----------
def insert_registration(row: dict):
    with engine.begin() as conn:
        cols = ",".join(row.keys()); vals = ",".join([f":{k}" for k in row.keys()])
        conn.execute(text(f"INSERT INTO registrations ({cols}) VALUES ({vals})"), row)

def update_registration(reg_id: str, updates: dict):
    placeholders = ", ".join([f"{k} = :{k}" for k in updates.keys()])
    updates["id"] = reg_id
    with engine.begin() as conn:
        conn.execute(text(f"UPDATE registrations SET {placeholders} WHERE id = :id"), updates)

def fetch_all(event_id: str | None = None):
    with engine.begin() as conn:
        if event_id:
            return pd.read_sql("SELECT * FROM registrations WHERE event_id = :e ORDER BY created_at DESC", conn, params={"e": event_id})
        return pd.read_sql("SELECT * FROM registrations ORDER BY created_at DESC", conn)

def fetch_distinct_academies(event_id: str | None = None):
    with engine.begin() as conn:
        try:
            if event_id:
                rows = conn.execute(text("""
                    SELECT DISTINCT academy FROM registrations 
                    WHERE academy IS NOT NULL AND academy <> '' AND event_id = :e
                """), {"e": event_id}).fetchall()
            else:
                rows = conn.execute(text("""
                    SELECT DISTINCT academy FROM registrations 
                    WHERE academy IS NOT NULL AND academy <> ''
                """)).fetchall()
            return sorted({r[0] for r in rows})
        except Exception:
            return []

def count_by_category(category_value: str, event_id: str | None = None):
    with engine.begin() as conn:
        try:
            if event_id:
                row = conn.execute(text("""
                    SELECT COUNT(*) FROM registrations WHERE category = :c AND event_id = :e
                """), {"c": category_value, "e": event_id}).fetchone()
            else:
                row = conn.execute(text("SELECT COUNT(*) FROM registrations WHERE category = :c"), {"c": category_value}).fetchone()
            return int(row[0] if row else 0)
        except Exception:
            return 0

# --------- [EVENTS] CRUD ----------
def insert_event(ev: dict):
    with engine.begin() as conn:
        cols = ",".join(ev.keys()); vals = ",".join([f":{k}" for k in ev.keys()])
        conn.execute(text(f"INSERT INTO events ({cols}) VALUES ({vals})"), ev)

def update_event(event_id: str, updates: dict):
    placeholders = ", ".join([f"{k} = :{k}" for k in updates.keys()])
    updates["id"] = event_id
    with engine.begin() as conn:
        conn.execute(text(f"UPDATE events SET {placeholders} WHERE id = :id"), updates)

def fetch_events(status: str | None = None):
    with engine.begin() as conn:
        if status:
            return pd.read_sql("SELECT * FROM events WHERE status = :s ORDER BY date ASC, created_at DESC", conn, params={"s": status})
        return pd.read_sql("SELECT * FROM events ORDER BY date ASC, created_at DESC", conn)

def get_event(event_id: str):
    with engine.begin() as conn:
        df = pd.read_sql("SELECT * FROM events WHERE id = :e", conn, params={"e": event_id})
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
# 7) OCR / VERIFICAÇÃO EID + UPLOAD
# ================================
EID_REGEX = re.compile(r"784-\d{4}-\d{7}-\d")
DATE_REGEXES = [re.compile(r"\b(\d{2})/(\d{2})/(\d{4})\b"), re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")]
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
    score = (0.3 if ratio_ok else
