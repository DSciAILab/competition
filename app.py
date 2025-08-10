# ================================
# app.py – ADSS Jiu-Jitsu Competition (Versão Complexa c/ Patch OCR)
# Mobile-first + máscaras live + OCR de EID (Google Vision opcional) + fallback local
# Patch: aplica dados de OCR via buffers no início da tela e usa st.rerun()
# Divisão preview, responsável de menor, peso atual, carteirinha JPG, Face ID, Admin
# ================================

# ---------- IMPORTS ----------
import os, re, uuid, urllib.parse, base64, json, math, datetime as dt
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
        # Se o JSON da credencial vier embutido (base64) nos secrets:
        if "GCP_CREDS_B64" in st.secrets:
            creds_b64 = st.secrets["GCP_CREDS_B64"]
            creds_json = base64.b64decode(creds_b64.encode()).decode()
            cred_path = "gcp_creds.json"
            with open(cred_path, "w") as f: f.write(creds_json)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
        # Ou use GOOGLE_APPLICATION_CREDENTIALS já configurado no ambiente
        vision_client = vision.ImageAnnotatorClient()
    except Exception as e:
        USE_VISION_OCR = False
        st.warning(f"OCR via Google Vision não pôde ser inicializado: {e}. Usando fallback local.")

# ================================
# 1) CONFIG GLOBAL (FULL SCREEN + MOBILE)
# ================================
st.set_page_config(page_title="ADSS Jiu-Jitsu Competition", layout="wide")

st.markdown("""
<style>
#MainMenu, header, footer { visibility: hidden; }
.block-container { padding-top: .6rem; padding-bottom: 3rem; max-width: 860px; }

/* Botões grandes do menu (2 linhas via \\n) */
.stButton > button {
  width: 100%; padding: 1rem 1.1rem; border-radius: 14px;
  font-weight: 700; text-align: left; white-space: pre-wrap; line-height: 1.25;
}

/* Inputs altos para toque */
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] { min-height: 44px; }

/* Câmera full-width */
[data-testid="stCameraInput"] video, [data-testid="stCameraInput"] img { width: 100% !important; height: auto !important; }

[data-testid="stDataFrame"] { border-radius: 12px; }
.header-sub { color:#9aa0a6; margin:.25rem 0 1rem; text-align:center; }
.section-gap { height: .6rem; }
.badge-pending{display:inline-block;padding:2px 8px;border-radius:999px;background:#F59E0B;color:#111;font-size:12px;font-weight:600;margin-left:8px;}
.small-muted {font-size:.9rem; color:#9aa0a6;}
.preview-box { border:1px solid #2a2a2a; border-radius:12px; padding:12px; background:#0f1116; }
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
if not ADMIN_PASSWORD:
    ADMIN_PASSWORD = "admin123"

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
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS registrations (
            id TEXT PRIMARY KEY,
            created_at TEXT,
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
            modality TEXT,        -- legado (não usado)
            belt TEXT,
            weight_class TEXT,    -- legado (não usado)
            weight_kg REAL,       -- peso atual do atleta
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
            "ALTER TABLE registrations ADD COLUMN face_phash TEXT"
        ]:
            try: conn.execute(text(alter))
            except Exception: pass
init_db()

def insert_registration(row: dict):
    with engine.begin() as conn:
        cols = ",".join(row.keys()); vals = ",".join([f":{k}" for k in row.keys()])
        conn.execute(text(f"INSERT INTO registrations ({cols}) VALUES ({vals})"), row)

def update_registration(reg_id: str, updates: dict):
    placeholders = ", ".join([f"{k} = :{k}" for k in updates.keys()])
    updates["id"] = reg_id
    with engine.begin() as conn:
        conn.execute(text(f"UPDATE registrations SET {placeholders} WHERE id = :id"), updates)

def fetch_all():
    with engine.begin() as conn:
        return pd.read_sql("SELECT * FROM registrations ORDER BY created_at DESC", conn)

def fetch_distinct_academies():
    with engine.begin() as conn:
        try:
            rows = conn.execute(text("SELECT DISTINCT academy FROM registrations WHERE academy IS NOT NULL AND academy <> ''")).fetchall()
            return sorted({r[0] for r in rows})
        except Exception:
            return []

def count_by_category(category_value: str):
    with engine.begin() as conn:
        try:
            row = conn.execute(text("SELECT COUNT(*) FROM registrations WHERE category = :c"), {"c": category_value}).fetchone()
            return int(row[0] if row else 0)
        except Exception:
            return 0

# ================================
# 4) UTIL – PIL helpers (textbbox compat)
# ================================
def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    """Compatível com Pillow moderno: usa textbbox para medir."""
    bbox = draw.textbbox((0,0), text, font=font)
    return (bbox[2]-bbox[0], bbox[3]-bbox[1])

# ================================
# 5) IMAGEM / FACE / PHASH / CARDS
# ================================
def save_image(file, reg_id: str, suffix: str, max_size=(800, 800)) -> str:
    if file is None: return ""
    image = Image.open(file); image.thumbnail(max_size)
    dest = UPLOAD_DIR / f"{reg_id}_{suffix}.jpg"
    image.save(dest, format="JPEG", quality=85)
    return str(dest)

def count_faces_in_image(file) -> int:
    if not FACE_DETECT_AVAILABLE: return -1
    if file is None: return 0
    data = file.getvalue()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(80,80))
    return 0 if faces is None else len(faces)

def crop_face_from_uploaded(file):
    if file is None: return None
    data = file.getvalue()
    img_pil = Image.open(BytesIO(data)).convert("RGB")
    if FACE_DETECT_AVAILABLE:
        arr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(80,80))
            if faces is not None and len(faces)>0:
                x,y,w,h = max(faces, key=lambda r:r[2]*r[3])
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(rgb).crop((x,y,x+w,y+h))
    return img_pil

def compute_phash(img_pil: Image.Image) -> str:
    if img_pil is None: return ""
    return str(imagehash.phash(img_pil))

def phash_distance(a: str, b: str) -> int:
    if not a or not b: return 1_000_000
    return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)

# ---- Cartão de inscrição (usa textbbox) ----
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

# ---- Carteirinha (membership) ----
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

# Telefone (05_-___-____)
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

# EID (784-####-#######-#)
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

# Data (dd/mm/aaaa)
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

# Opções
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

def to_text_lines(s: str):
    return [l.strip() for l in s.splitlines() if l.strip()]

def google_vision_ocr_extract(image_bytes: bytes) -> dict:
    """Usa Vision para retornar {text, eid, dob, nationality, name, confidence, keywords_found}"""
    out = {"text": "", "eid": "", "dob": "", "nationality": "", "name": "", "confidence": 0.0, "keywords_found": []}
    if not USE_VISION_OCR:
        return out
    try:
        image = vision.Image(content=image_bytes)
        resp = vision_client.document_text_detection(image=image)
        if resp.error.message:
            return out
        full_text = resp.full_text_annotation.text or ""
        out["text"] = full_text
        text_lower = full_text.lower()
        out["keywords_found"] = [kw for kw in KEYWORDS_EID if kw in text_lower]

        # EID
        m = EID_REGEX.search(full_text.replace(" ", ""))
        if m: out["eid"] = m.group(0)

        # Data de nascimento
        for rx in DATE_REGEXES:
            m = rx.search(full_text)
            if m:
                try:
                    if rx.pattern.startswith(r"\b(\d{2})/"):  # dd/mm/yyyy
                        d,mn,y = map(int, m.groups())
                    else:  # yyyy-mm-dd
                        y,mn,d = map(int, m.groups())
                    dob = dt.date(y,mn,d)
                    out["dob"] = f"{d:02d}/{mn:02d}/{y:04d}"
                    break
                except Exception:
                    pass

        # Nacionalidade
        lines = to_text_lines(full_text)
        for li in lines:
            low = li.lower()
            if any(k in low for k in NATIONALITY_KEYS):
                parts = re.split(r"[:\-]", li, maxsplit=1)
                cand = parts[1].strip() if len(parts) > 1 else li
                out["nationality"] = squeeze_spaces(re.sub(r"(?i)nationality", "", cand)).strip()
                break
        # Nome
        for li in lines:
            low = li.lower()
            if any(k in low for k in NAME_KEYS):
                parts = re.split(r"[:\-]", li, maxsplit=1)
                cand = parts[1].strip() if len(parts) > 1 else li
                cand = re.sub(r"(?i)name|cardholder|card holder|full name|nome|الاسم", "", cand).strip()
                out["name"] = squeeze_spaces(cand)
                break

        # confiança simples (heurística): 0-1
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
    """Fallback local: mantém leve; sem OCR adicional (pode plugar easyocr aqui se quiser)."""
    out = {"text": "", "eid": "", "dob": "", "nationality": "", "name": "", "confidence": 0.0, "keywords_found": []}
    return out

def is_likely_eid_card(image_bytes: bytes, ocr_text: str) -> (bool, float, dict):
    """Combina proporção + regex + keywords para classificar imagem como EID."""
    details = {}
    try:
        img = Image.open(BytesIO(image_bytes))
        w, h = img.size
        ratio = w / h if h else 0
        details["ratio"] = ratio
        ratio_ok = 1.427 <= ratio <= 1.745   # ~1.586 ± ~10%
    except Exception:
        ratio_ok = False

    ocr_low = (ocr_text or "").lower()
    eid_found = bool(EID_REGEX.search((ocr_text or "").replace(" ", "")))
    kw_found = any(kw in ocr_low for kw in KEYWORDS_EID)

    score = 0.0
    if ratio_ok: score += 0.3
    if eid_found: score += 0.5
    if kw_found: score += 0.2
    return (score >= 0.6), score, details

def extract_eid_fields_from_image(uploaded_file, for_guardian=False):
    """Pipeline: lê imagem, roda OCR (Vision se ativo, senão fallback), valida EID e retorna dict com campos."""
    if uploaded_file is None:
        return {"ok": False, "reason": "Sem imagem.", "data": {}}

    img_bytes = uploaded_file.getvalue()
    ocr = google_vision_ocr_extract(img_bytes) if USE_VISION_OCR else local_heuristic_extract(img_bytes)
    likely, score, geo = is_likely_eid_card(img_bytes, ocr.get("text",""))

    result = {
        "ok": likely,
        "reason": f"confiança={score:.2f}, ratio={geo.get('ratio','?')}",
        "data": {
            "eid": eid_format_live(ocr.get("eid","")),
            "dob": ocr.get("dob",""),
            "nationality": ocr.get("nationality",""),
            "full_name": ocr.get("name",""),
        }
    }
    return result

# ================================
# 8) ESTADO / NAVEGAÇÃO
# ================================
if "screen" not in st.session_state: st.session_state["screen"]="welcome"
if "accepted_terms" not in st.session_state: st.session_state["accepted_terms"]=False
if "event_name" not in st.session_state: st.session_state["event_name"]=EVENT_DEFAULT_NAME
if "region" not in st.session_state: st.session_state["region"]=EVENT_DEFAULT_REGION
if "errors" not in st.session_state: st.session_state["errors"]=set()

def whatsapp_button(phone: str):
    if not phone or not phone.strip(): return
    phone_clean = phone.replace("+","").replace(" ","").replace("-","")
    url = f"https://wa.me/{phone_clean}?text=" + urllib.parse.quote(f"Olá! Tenho uma dúvida sobre o evento {st.session_state['event_name']}.")
    st.markdown(
        f"""<a href="{url}" target="_blank"
           style="position:fixed;right:20px;bottom:20px;background:#25D366;color:#fff;
           padding:12px 16px;border-radius:999px;text-decoration:none;font-weight:700;
           box-shadow:0 4px 16px rgba(0,0,0,.2);z-index:9999;">WhatsApp da Organização</a>""",
        unsafe_allow_html=True
    )

# ================================
# 9) TELAS
# ================================
def screen_welcome():
    st.markdown(f"<h1 style='text-align:center;margin:0'>{st.session_state['event_name']}</h1>", unsafe_allow_html=True)
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
    cols = st.columns(2)
    if cols[0].button("Prosseguir"):
        if agree:
            st.session_state["accepted_terms"]=True; st.session_state["screen"]="menu"
        else:
            st.error("Você precisa concordar com os termos e condições para continuar.")

def screen_menu():
    st.markdown(f"<h1 style='text-align:center;margin:0'>{st.session_state['event_name']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='header-sub'>{st.session_state['region']}</p>", unsafe_allow_html=True)
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

def screen_public_list():
    st.title("Lista de Inscritos (pública)")
    df = fetch_all()
    if df.empty:
        st.info("Ainda não há inscrições.")
    else:
        show = df[["id","first_name","last_name","academy","category","belt","weight_kg","approval_status"]].copy()
        show["Nome"] = (show["first_name"].fillna("") + " " + show["last_name"].fillna("")).str.strip()
        show = show.rename(columns={
            "id":"Inscrição","academy":"Academia","category":"Categoria",
            "belt":"Faixa","weight_kg":"Peso (kg)","approval_status":"Status"
        })[["Inscrição","Nome","Academia","Categoria","Faixa","Peso (kg)","Status"]]
        st.dataframe(show, use_container_width=True, height=420)
    st.markdown("---")
    st.button("Voltar ao menu principal", on_click=lambda: st.session_state.update(screen="menu"))

def stats_view(embed: bool=False):
    df = fetch_all()
    if df.empty:
        st.info("Ainda não há inscrições."); return
    totals = (df.groupby("category", dropna=False).size().reset_index(name="inscritos")
                .sort_values("inscritos", ascending=False).reset_index(drop=True))
    if not embed:
        st.title("Estatísticas – Inscritos por Categoria")
    st.subheader("Total por categoria")
    st.dataframe(totals, use_container_width=True)
    st.bar_chart(totals.set_index("category"))
    if {"category","belt"}.issubset(df.columns):
        st.subheader("Por faixa")
        st.dataframe(df.pivot_table(index="category", columns="belt", values="id", aggfunc="count", fill_value=0), use_container_width=True)

def admin_view():
    st.subheader("Painel do Organizador")
    df = fetch_all()
    st.write(f"Total de inscrições: {len(df)}")
    if df.empty:
        st.info("Ainda não há inscrições."); return
    for _, row in df.iterrows():
        name = f"{row.get('first_name','')} {row.get('last_name','')}".strip()
        status = (row.get("approval_status") or "Pending").strip()
        badge = "<span class='badge-pending'>Pendente de aprovação</span>" if status.lower()=="pending" else ""
        header = (name or row.get('id','(sem id)')) + " " + badge
        with st.expander(header, expanded=False):
            meta = {
                "ID": row.get("id",""),
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
            if row.get("id_doc_photo_path") and Path(row["id_doc_photo_path"]).exists():
                st.image(row["id_doc_photo_path"], caption="Documento do atleta", width=220)
            if row.get("guardian_eid_photo_path") and Path(row["guardian_eid_photo_path"]).exists():
                st.image(row["guardian_eid_photo_path"], caption="EID do responsável", width=220)

            jpg = generate_registration_jpg(row, st.session_state["event_name"], st.session_state["region"])
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
        if admin_pw == ADMIN_PASSWORD:
            st.session_state["admin_ok"]=True
        else:
            st.error("Senha incorreta.")
    if st.session_state.get("admin_ok"):
        admin_view()
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
        new_profile = st.camera_input("Nova foto de perfil") or st.file_uploader("Ou enviar arquivo", type=["jpg","jpeg","png"], key=f"file_{best_row['id']}")
        submitted_update = st.form_submit_button("Salvar alterações")

    try:
        mem_jpg = generate_membership_card(best_row.get("profile_photo_path",""), best_row.get("id",""),
                                           best_row.get("eid",""), best_row.get("age_years",0),
                                           f"{best_row.get('first_name','')} {best_row.get('last_name','')}".strip(),
                                           best_row.get("belt",""))
        st.download_button("Baixar carteirinha (JPG)", data=mem_jpg,
                           file_name=f"membership_{best_row['id']}.jpg", mime="image/jpeg",
                           key=f"dl_edit_member_{best_row['id']}")
    except Exception:
        pass

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
    df = fetch_all()
    if df.empty:
        st.info("Não há inscrições para alterar.")
        st.button("Voltar ao menu principal", on_click=lambda: st.session_state.update(screen="menu")); return
    tab1, tab2 = st.tabs(["Face ID (selfie)", "ID da inscrição"])
    with tab1:
        st.write("Envie uma selfie recente do atleta.")
        qfile = st.camera_input("Tirar selfie") or st.file_uploader("Ou enviar arquivo", type=["jpg","jpeg","png"], key="upload_edit")
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
                st.error("ID não encontrado.")
            else:
                render_edit_form(row.iloc[0])
    st.markdown("---"); st.button("Voltar ao menu principal", on_click=lambda: st.session_state.update(screen="menu"))

def screen_new_registration():
    # Máscaras live
    for k in ["phone","coach_phone","guardian_phone"]:
        if k in st.session_state: st.session_state[k] = format_phone_live(st.session_state[k])
    for k in ["eid","guardian_eid"]:
        if k in st.session_state: st.session_state[k] = eid_format_live(st.session_state[k])
    if "dob" in st.session_state: st.session_state["dob"] = date_mask_live(st.session_state["dob"])

    # --- APLICAÇÃO DE DADOS DE OCR (ANTES DE CRIAR OS WIDGETS) ---
    if st.session_state.get("ocr_apply_atleta"):
        data = st.session_state.pop("ocr_apply_atleta")
        if data.get("eid"):
            st.session_state["eid"] = eid_format_live(data["eid"])
        if data.get("dob"):
            st.session_state["dob"] = date_mask_live(data["dob"])
        if data.get("nationality"):
            st.session_state["nationality"] = data["nationality"]
        if data.get("full_name"):
            tokens = data["full_name"].split()
            if tokens:
                st.session_state["first_name"] = title_capitalize(tokens[0])
                st.session_state["last_name"]  = title_capitalize(" ".join(tokens[1:])) if len(tokens)>1 else st.session_state.get("last_name","")

    if st.session_state.get("ocr_apply_guardian"):
        data = st.session_state.pop("ocr_apply_guardian")
        if data.get("eid"):
            st.session_state["guardian_eid"] = eid_format_live(data["eid"])
        if data.get("full_name"):
            st.session_state["guardian_name"] = title_capitalize(data["full_name"])

    st.title("Formulário de Inscrição")
    st.caption("Preencha seus dados. Campos com * são obrigatórios.")

    # --- PESSOAIS ---
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
        # idade automática
        dob_date = parse_masked_date(st.session_state.get("dob",""))
        age_years = compute_age_year_based(dob_date.year) if dob_date else 0
        st.text_input("Idade (auto)", value=str(age_years if age_years else ""), disabled=True)

    # --- ACADEMIA / COACH ---
    st.subheader("Academia")
    academies = fetch_distinct_academies()
    options = ["Selecione...", "Minha academia não está na lista"] + academies
    academy_choice = st.selectbox("Academia*", options, key="academy_choice", index=st.session_state.get("academy_choice_idx", 0))
    academy_other = ""
    if academy_choice == "Minha academia não está na lista":
        academy_other = st.text_input("Nome da academia*", key="academy_other", placeholder="Nome da academia")
    coach      = st.text_input("Professor/Coach", key="coach", placeholder="Nome do coach")
    coach_phone= st.text_input("Telefone do Professor (05_-___-____)", key="coach_phone", placeholder="052-123-4567")

    # --- FOTOS ---
    st.subheader("Documentos")
    profile_img = st.camera_input("Foto de Perfil (rosto visível)*", key="profile_img")
    id_doc_img  = st.camera_input("Documento de Identificação do atleta (frente)*", key="id_doc_img")

    # Botão para ler dados da EID (atleta) – usa buffer + rerun
    colx, _ = st.columns([1,4])
    with colx:
        if st.button("Ler dados da EID do atleta"):
            res = extract_eid_fields_from_image(id_doc_img)
            if not res["ok"]:
                st.warning(f"Foto pode não ser uma EID ({res['reason']}). Mesmo assim tentando aproveitar dados extraídos.")
            st.session_state["ocr_apply_atleta"] = res.get("data", {})
            st.rerun()

    # --- COMPETIÇÃO (sem modalidade) ---
    st.subheader("Informações de Competição")
    belt        = st.selectbox("Faixa*", FAIXAS_GI, key="belt")
    weight_kg   = st.number_input("Peso atual do atleta (kg)*", min_value=0.0, step=0.1, key="weight_kg")

    # --- RESPONSÁVEL (menor de 18) ---
    if age_years and age_years < 18:
        st.subheader("Responsável legal (obrigatório para menor de 18)")
        guardian_name  = st.text_input("Nome completo do responsável*", key="guardian_name")
        guardian_eid   = st.text_input("EID do responsável (784-####-#######-#)*", key="guardian_eid")
        guardian_phone = st.text_input("Telefone do responsável (05_-___-____)*", key="guardian_phone")
        guardian_eid_photo = st.camera_input("Foto da EID do responsável (frente)*", key="guardian_eid_photo")

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

    # --- DIVISÃO PERFEITA (preview) ---
    if all([gender, belt, weight_kg, age_years]):
        age_div = age_division_by_year(age_years)
        st.markdown("<div class='preview-box'>", unsafe_allow_html=True)
        st.markdown(f"**Divisão perfeita:** {gender} / {age_div} / {belt} / {weight_kg:.1f} kg")
        st.caption("Revise antes de enviar.")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- CONSENTIMENTO + ENVIAR ---
    st.subheader("Termo de Consentimento")
    consent = st.checkbox("Eu li e concordo com o termo de consentimento*", key="consent")

    submitted = st.button("Enviar inscrição")

    if submitted:
        errors = set()
        must = ["first_name","last_name","email","phone","gender","nationality","belt","weight_kg","consent","dob","profile_img","id_doc_img","academy_choice","eid"]
        for k in must:
            if not st.session_state.get(k): errors.add(k)

        if st.session_state.get("academy_choice") == "Minha academia não está na lista" and not st.session_state.get("academy_other"):
            errors.add("academy_other")

        # validações específicas
        if st.session_state.get("eid") and not eid_is_valid(st.session_state["eid"]):
            errors.add("eid"); st.error("EID do atleta inválido. Use o formato 784-####-#######-#.")
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

        # menor de idade
        if age_years and age_years < 18:
            if not st.session_state.get("guardian_name"): errors.add("guardian_name")
            if not st.session_state.get("guardian_eid") or not eid_is_valid(st.session_state.get("guardian_eid","")):
                errors.add("guardian_eid"); st.error("EID do responsável inválido.")
            if not st.session_state.get("guardian_phone"): errors.add("guardian_phone")
            if st.session_state.get("guardian_eid_photo") is None:
                errors.add("guardian_eid_photo")

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

        # salvar imagens
        reg_id = str(uuid.uuid4())[:8].upper()
        profile_photo_path = save_image(profile_img, reg_id, "profile")
        id_doc_photo_path = save_image(id_doc_img, reg_id, "id_doc")
        guardian_eid_photo_path = save_image(guardian_eid_photo, reg_id, "guardian_eid") if guardian_eid_photo else ""
        face_crop_img = crop_face_from_uploaded(profile_img)
        face_phash = compute_phash(face_crop_img)

        row = {
            "id": reg_id, "created_at": dt.datetime.utcnow().isoformat(),
            "event_name": st.session_state["event_name"],
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
            st.info(f"Atletas já inscritos na categoria '{category}': {count_by_category(category)}")

            # Cartão + Carteirinha
            jpg_bytes = generate_registration_jpg(row, st.session_state["event_name"], st.session_state["region"])
            st.download_button("Baixar inscrição (JPG)", data=jpg_bytes, file_name=f"inscricao_{row['id']}.jpg", mime="image/jpeg")

            try:
                member_card = generate_membership_card(profile_photo_path, reg_id, eid_masked, age_years,
                                                       f"{first_name} {last_name}", st.session_state["belt"])
                st.download_button("Baixar carteirinha (JPG)", data=member_card, file_name=f"membership_{reg_id}.jpg", mime="image/jpeg")
                st.image(member_card, caption="Pré-visualização da carteirinha", use_column_width=True)
            except Exception as e:
                st.warning(f"Não foi possível gerar a carteirinha: {e}")

            if WHATSAPP_PHONE.strip():
                msg = f"Olá! Minha inscrição do evento {st.session_state['event_name']} foi enviada. Meu ID é {reg_id}."
                phone_clean = WHATSAPP_PHONE.replace("+", "").replace(" ", "").replace("-", "")
                wa_url = f"https://wa.me/{phone_clean}?text=" + urllib.parse.quote(msg)
                st.markdown(f"[Falar com a organização no WhatsApp]({wa_url})")
        except Exception as e:
            st.error(f"Erro ao salvar inscrição: {e}")

    st.markdown("---"); st.button("Voltar ao menu principal", on_click=lambda: st.session_state.update(screen="menu"))

# ================================
# 10) ROTEAMENTO + WHATSAPP
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
else:
    st.session_state["screen"] = "menu"; screen_menu()
