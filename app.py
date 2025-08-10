# ================================
# app.py – ADSS Jiu-Jitsu Competition
# Mobile-first + menu minimalista + máscaras em tempo real (telefone/EID/data)
# Idade automática, peso atual, sem modalidade, carteirinha (JPG)
# ================================

# ---------- IMPORTS ----------
import os, re, uuid, urllib.parse, datetime as dt
from pathlib import Path
from io import BytesIO

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from PIL import Image

import numpy as np
try:
    import cv2
    FACE_DETECT_AVAILABLE = True
except Exception:
    cv2 = None
    FACE_DETECT_AVAILABLE = False

import imagehash

# ================================
# 1) CONFIG GLOBAL (FULL SCREEN + MOBILE)
# ================================
st.set_page_config(page_title="ADSS Jiu-Jitsu Competition", layout="wide")

st.markdown("""
<style>
#MainMenu, header, footer { visibility: hidden; }
.block-container { padding-top: .6rem; padding-bottom: 3rem; max-width: 860px; }

.stButton > button {
  width: 100%; padding: 1rem 1.1rem; border-radius: 14px;
  font-weight: 700; text-align: left; white-space: pre-wrap; line-height: 1.25;
}

.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] { min-height: 44px; }

[data-testid="stCameraInput"] video, [data-testid="stCameraInput"] img { width: 100% !important; height: auto !important; }

[data-testid="stDataFrame"] { border-radius: 12px; }
.header-sub { color:#9aa0a6; margin:.25rem 0 1rem; text-align:center; }
.section-gap { height: .6rem; }
.badge-pending{display:inline-block;padding:2px 8px;border-radius:999px;background:#F59E0B;color:#111;font-size:12px;font-weight:600;margin-left:8px;}
.small-muted {font-size:.9rem; color:#9aa0a6;}
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
            -- modality removida do fluxo; coluna pode permanecer vazia
            modality TEXT,
            belt TEXT,
            weight_class TEXT,   -- legado
            weight_kg REAL,      -- peso atual do atleta
            category TEXT,
            consent INTEGER,
            profile_photo_path TEXT,
            id_doc_photo_path TEXT,
            approval_status TEXT,
            face_phash TEXT
        );
        """))
        # migrações leves
        for alter in [
            "ALTER TABLE registrations ADD COLUMN eid TEXT",
            "ALTER TABLE registrations ADD COLUMN weight_kg REAL",
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
# 4) IMAGEM / FACE / PHASH / CARDS
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

# ---- Cartão de inscrição (com EID e sem modalidade) ----
def generate_registration_jpg(row: dict, event_name: str, region: str, logo_file=None, width=1080, height=1350) -> bytes:
    from PIL import ImageDraw, ImageFont
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
        except Exception: pass

    title_text = event_name or "ADSS Jiu-Jitsu Competition"
    w_title, _ = draw.textsize(title_text, font=font_title)
    draw.text(((width - w_title) / 2, y), title_text, fill=(0, 0, 0), font=font_title); y += 70
    if region:
        w_reg, _ = draw.textsize(region, font=font_sub)
        draw.text(((width - w_reg) / 2, y), region, fill=(80, 80, 80), font=font_sub); y += 50

    draw.line([(80, y), (width - 80, y)], fill=(230,230,230), width=3); y += 30

    x_left = 80; photo_size = 320
    profile_path = row.get("profile_photo_path", "")
    if profile_path and os.path.exists(profile_path):
        try:
            p = Image.open(profile_path).convert("RGB").resize((photo_size, photo_size))
            img.paste(p, (x_left, y))
        except Exception: pass

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
    line("Peso atual:", f"{row.get('weight_kg','-')} kg" if row.get("weight_kg") else "-")
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
    except Exception: pass

    buf = BytesIO(); img.save(buf, format="JPEG", quality=90, optimize=True); return buf.getvalue()

# ---- Carteirinha (membership) ----
def generate_membership_card(img_profile_path: str, membership_id: str, eid: str, age: int, name: str, belt: str, width=900, height=600) -> bytes:
    """Gera a carteirinha em JPG com: foto, Membership (ID), EID, idade, nome, faixa e QR (membership)."""
    from PIL import ImageDraw, ImageFont
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
        except Exception: pass

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
# 5) NORMALIZAÇÃO / MÁSCARAS / OPÇÕES
# ================================
def squeeze_spaces(s:str)->str: return re.sub(r"\s+", " ", s or "").strip()
def title_capitalize(s:str)->str: return " ".join([w.capitalize() for w in squeeze_spaces(s).split(" ")])
def normalize_academy(s:str)->str: return title_capitalize(s)

# Telefone (05_-___-____) – máscara live e final
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

# EID (784-####-#######-#) – máscara live e validação
def eid_format_live(raw: str) -> str:
    digits = re.sub(r"\D", "", raw or "")
    digits = digits[:15]
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

# Data (dd/mm/aaaa) – máscara live e parse
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
# 6) ESTADO / CALLBACKS LIVE
# ================================
if "screen" not in st.session_state: st.session_state["screen"]="welcome"
if "accepted_terms" not in st.session_state: st.session_state["accepted_terms"]=False
if "event_name" not in st.session_state: st.session_state["event_name"]=EVENT_DEFAULT_NAME
if "region" not in st.session_state: st.session_state["region"]=EVENT_DEFAULT_REGION
if "errors" not in st.session_state: st.session_state["errors"]=set()

# buffers live
for k in ["phone_live","eid_live","dob_live","age_live"]:
    if k not in st.session_state: st.session_state[k] = ""

def cb_phone():
    st.session_state["phone_live"] = format_phone_live(st.session_state.get("phone_raw",""))

def cb_eid():
    st.session_state["eid_live"] = eid_format_live(st.session_state.get("eid_raw",""))

def cb_dob():
    st.session_state["dob_live"] = date_mask_live(st.session_state.get("dob_raw",""))
    d = parse_masked_date(st.session_state["dob_live"])
    st.session_state["age_live"] = str(compute_age_year_based(d.year)) if d else ""

def label(label_text: str, key: str
