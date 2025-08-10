# ================================
# app.py – ADSS Jiu-Jitsu Competition (VERSÃO REVISADA)
# Mobile-first + menu minimalista + máscaras em tempo real (telefone/EID/data)
# Idade automática, peso atual, sem modalidade, carteirinha (JPG)
# ================================

# ---------- IMPORTS ----------
import os, re, uuid, urllib.parse, datetime as dt
from pathlib import Path
from io import BytesIO

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text, insert, update
from PIL import Image

import numpy as np
try:
    import cv2
    FACE_DETECT_AVAILABLE = True
except ImportError:
    cv2 = None
    FACE_DETECT_AVAILABLE = False

try:
    import qrcode
    QRCODE_AVAILABLE = True
except ImportError:
    qrcode = None
    QRCODE_AVAILABLE = False

import imagehash

# ================================
# 1) CONFIG GLOBAL (FULL SCREEN + MOBILE)
# ================================
st.set_page_config(page_title="ADSS Jiu-Jitsu Competition", layout="wide")

## SUGESTÃO: Para melhor organização, mova este CSS para um arquivo `style.css`
## e carregue-o com a função abaixo.
# def load_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# load_css("style.css")

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
## REVISADO: Removida senha padrão "hardcoded" para maior segurança.
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD") or st.secrets.get("ADMIN_PASSWORD")
if not ADMIN_PASSWORD:
    # Se nenhuma senha for definida, a funcionalidade de admin será desativada.
    st.toast("AVISO: Senha de admin não configurada.", icon="⚠️")

WHATSAPP_PHONE = os.getenv("WHATSAPP_ORG", "+9715xxxxxxxx")

DB_PATH = "registrations.db"
UPLOAD_DIR = Path("uploads"); UPLOAD_DIR.mkdir(exist_ok=True)

EVENT_DEFAULT_NAME = "ADSS Jiu-Jitsu Competition"
EVENT_DEFAULT_REGION = "Al Dhafra – Abu Dhabi"

# ================================
# 3) BANCO DE DADOS (SQLite)
# ================================
engine = create_engine(f"sqlite:///{DB_PATH}", future=True)

## REVISADO: Sistema de versionamento para aplicar migrações de forma segura.
def init_db():
    migrations = {
        1: "ALTER TABLE registrations ADD COLUMN eid TEXT",
        2: "ALTER TABLE registrations ADD COLUMN weight_kg REAL",
        3: "ALTER TABLE registrations ADD COLUMN first_name TEXT",
        4: "ALTER TABLE registrations ADD COLUMN last_name TEXT",
        5: "ALTER TABLE registrations ADD COLUMN full_name_en TEXT",
        6: "ALTER TABLE registrations ADD COLUMN coach_phone TEXT",
        7: "ALTER TABLE registrations ADD COLUMN age_years INTEGER",
        8: "ALTER TABLE registrations ADD COLUMN approval_status TEXT",
        9: "ALTER TABLE registrations ADD COLUMN face_phash TEXT",
    }

    with engine.begin() as conn:
        # Tabela principal
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS registrations (
            id TEXT PRIMARY KEY, created_at TEXT, event_name TEXT, first_name TEXT,
            last_name TEXT, full_name_en TEXT, email TEXT, phone TEXT, eid TEXT,
            nationality TEXT, gender TEXT, dob TEXT, age_years INTEGER,
            age_division TEXT, academy TEXT, coach TEXT, coach_phone TEXT,
            region TEXT, modality TEXT, belt TEXT, weight_class TEXT,
            weight_kg REAL, category TEXT, consent INTEGER,
            profile_photo_path TEXT, id_doc_photo_path TEXT,
            approval_status TEXT, face_phash TEXT
        );"""))
        
        # Tabela de versionamento
        conn.execute(text("CREATE TABLE IF NOT EXISTS db_version (version INTEGER)"))
        
        # Obter versão atual
        res = conn.execute(text("SELECT version FROM db_version")).fetchone()
        current_version = res[0] if res else 0
        if not res:
            conn.execute(text("INSERT INTO db_version (version) VALUES (0)"))

        # Aplicar migrações pendentes
        for v in sorted(migrations.keys()):
            if v > current_version:
                try:
                    conn.execute(text(migrations[v]))
                    conn.execute(text("UPDATE db_version SET version = :v"), {"v": v})
                    print(f"DB Migration applied: Version {v}")
                except Exception as e:
                    print(f"ERROR applying migration version {v}: {e}")
                    break # Para a execução se uma migração falhar
init_db()

## REVISADO: Uso do SQLAlchemy Core para construção segura da query.
def insert_registration(row: dict):
    with engine.begin() as conn:
        stmt = insert(text("registrations")).values(**row)
        conn.execute(stmt)

## REVISADO: Uso do SQLAlchemy Core para construção segura da query.
def update_registration(reg_id: str, updates: dict):
    # Garante que o ID não esteja no dicionário de `updates` para evitar conflito
    updates.pop('id', None)
    stmt = (
        update(text("registrations"))
        .where(text("id = :id"))
        .values(**updates)
    )
    with engine.begin() as conn:
        conn.execute(stmt, {"id": reg_id})

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
## REVISADO: Removidos "números mágicos", usando constantes para o layout.
def generate_registration_jpg(row: dict, event_name: str, region: str, logo_file=None, width=1080, height=1350) -> bytes:
    from PIL import ImageDraw, ImageFont

    # --- Constantes de Layout ---
    MARGIN, TOP_PAD = 80, 40
    PHOTO_SIZE, PHOTO_TEXT_GAP = 320, 40
    FONT_SIZE_TITLE, FONT_SIZE_SUB = 60, 32
    FONT_SIZE_BOLD, FONT_SIZE_TEXT = 36, 34
    LINE_H_LABEL, LINE_H_VALUE = 40, 54
    QR_SIZE = 240

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", FONT_SIZE_TITLE)
        font_sub   = ImageFont.truetype("DejaVuSans.ttf", FONT_SIZE_SUB)
        font_bold  = ImageFont.truetype("DejaVuSans-Bold.ttf", FONT_SIZE_BOLD)
        font_text  = ImageFont.truetype("DejaVuSans.ttf", FONT_SIZE_TEXT)
    except Exception:
        font_title = font_sub = font_bold = font_text = ImageFont.load_default()

    y = TOP_PAD
    if logo_file:
        try:
            lf = Image.open(logo_file).convert("RGBA")
            maxw = int(width * 0.5); ratio = maxw / lf.width
            lf = lf.resize((maxw, int(lf.height * ratio)))
            img.paste(lf, (int((width - lf.width) / 2), y), lf); y += lf.height + 20
        except Exception as e: print(f"Error loading logo: {e}")

    title_text = event_name or "ADSS Jiu-Jitsu Competition"
    w_title, _ = draw.textsize(title_text, font=font_title)
    draw.text(((width - w_title) / 2, y), title_text, fill=(0, 0, 0), font=font_title); y += 70
    if region:
        w_reg, _ = draw.textsize(region, font=font_sub)
        draw.text(((width - w_reg) / 2, y), region, fill=(80, 80, 80), font=font_sub); y += 50

    draw.line([(MARGIN, y), (width - MARGIN, y)], fill=(230,230,230), width=3); y += 30

    photo_y_start = y
    x_left = MARGIN
    if (path := row.get("profile_photo_path")) and os.path.exists(path):
        try:
            p = Image.open(path).convert("RGB").resize((PHOTO_SIZE, PHOTO_SIZE))
            img.paste(p, (x_left, y))
        except Exception as e: print(f"Error pasting profile photo: {e}")

    x_text = x_left + PHOTO_SIZE + PHOTO_TEXT_GAP
    def line(label, value):
        nonlocal y
        draw.text((x_text, y), label, fill=(0,0,0), font=font_bold); y += LINE_H_LABEL
        draw.text((x_text, y), value, fill=(40,40,40), font=font_text); y += LINE_H_VALUE

    name = f"{row.get('first_name','')} {row.get('last_name','')}".strip()
    line("Inscrição (ID):", row.get("id",""))
    line("Nome:", name or "-")
    line("EID:", row.get("eid","-") or "-")
    line("Categoria:", row.get("category","-") or "-")
    line("Faixa:", row.get("belt","-") or "-")
    line("Peso atual:", f"{row.get('weight_kg','-')} kg" if row.get('weight_kg') else "-")
    line("Nacionalidade:", row.get("nationality","-") or "-")

    y = max(y, photo_y_start + PHOTO_SIZE + 30)
    draw.line([(MARGIN, y), (width - MARGIN, y)], fill=(230,230,230), width=3); y += 30

    def kv(k, v):
        nonlocal y
        draw.text((MARGIN, y), k, fill=(0,0,0), font=font_bold); y += LINE_H_LABEL
        draw.text((MARGIN, y), v, fill=(40,40,40), font=font_text); y += LINE_H_VALUE

    kv_y_start = y
    kv("E-mail:", row.get("email","-") or "-")
    kv("Telefone:", row.get("phone","-") or "-")
    kv("Professor:", row.get("coach","-") or "-")
    kv("Status:", row.get("approval_status","Pending"))

    if QRCODE_AVAILABLE:
        try:
            qr_data = f"Inscrição: {row.get('id','')}\nEvento: {event_name}"
            qr_img = qrcode.make(qr_data).resize((QR_SIZE, QR_SIZE))
            img.paste(qr_img, (width - MARGIN - QR_SIZE, kv_y_start))
        except Exception as e: print(f"Error generating QR code: {e}")

    buf = BytesIO(); img.save(buf, format="JPEG", quality=90, optimize=True); return buf.getvalue()

# ---- Carteirinha (membership) ----
## REVISADO: Removidos "números mágicos", usando constantes para o layout.
def generate_membership_card(img_profile_path: str, membership_id: str, eid: str, age: int, name: str, belt: str, width=900, height=600) -> bytes:
    from PIL import ImageDraw, ImageFont

    # --- Constantes de Layout ---
    PAD = 30
    PHOTO_SIZE = 220
    BORDER = 10
    FONT_SIZE_TITLE, FONT_SIZE_TEXT, FONT_SIZE_BOLD = 42, 28, 30
    LINE_H_LABEL, LINE_H_VALUE = 36, 46
    QR_SIZE = 200

    canvas = Image.new("RGB", (width, height), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([(BORDER,BORDER),(width-BORDER,height-BORDER)], outline=(200,200,200), width=3)
    try:
        f_title = ImageFont.truetype("DejaVuSans-Bold.ttf", FONT_SIZE_TITLE)
        f_text  = ImageFont.truetype("DejaVuSans.ttf", FONT_SIZE_TEXT)
        f_bold  = ImageFont.truetype("DejaVuSans-Bold.ttf", FONT_SIZE_BOLD)
    except Exception:
        f_title = f_text = f_bold = ImageFont.load_default()

    y_photo = PAD + 40
    if img_profile_path and os.path.exists(img_profile_path):
        try:
            p = Image.open(img_profile_path).convert("RGB").resize((PHOTO_SIZE, PHOTO_SIZE))
            canvas.paste(p, (PAD, y_photo))
        except Exception as e: print(f"Error pasting profile photo for card: {e}")

    draw.text((PAD, 20), "ADSS Jiu-Jitsu – Membership", fill=(0,0,0), font=f_title)

    x = PAD + PHOTO_SIZE + PAD; y = y_photo
    def field(k,v):
        nonlocal y
        draw.text((x, y), k, fill=(0,0,0), font=f_bold); y += LINE_H_LABEL
        draw.text((x, y), v, fill=(40,40,40), font=f_text); y += LINE_H_VALUE
    field("Membership ID:", membership_id or "—")
    field("Nome:", name or "—")
    field("EID:", eid or "—")
    field("Idade:", f"{age} anos" if age else "—")
    field("Faixa:", belt or "—")

    if QRCODE_AVAILABLE:
        try:
            qr = qrcode.make(membership_id or "").resize((QR_SIZE,QR_SIZE))
            canvas.paste(qr, (width-PAD-QR_SIZE, height-PAD-QR_SIZE))
        except Exception as e:
            print(f"Error generating membership QR code: {e}")

    buf = BytesIO(); canvas.save(buf, format="JPEG", quality=92); return buf.getvalue()

# ================================
# 5) NORMALIZAÇÃO / MÁSCARAS / OPÇÕES
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
    digits = re.sub(r"\D", "", raw or "")
    if not digits.startswith("784"): digits = "784" + digits
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
    except (ValueError, TypeError):
        return None

# Opções
FAIXAS_GI = ["Branca","Cinza (Kids)","Amarela (Kids)","Laranja (Kids)","Verde (Kids)","Azul","Roxa","Marrom","Preta"]

def get_country_list():
    try:
        import pycountry
        countries = [c.name for c in pycountry.countries]
        extras = ["Hong Kong","Macau","Palestine","Kosovo"]
        return sorted(set(countries+extras))
    except ImportError:
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

def label(label_text: str, key: str):
    # (O restante do código da interface do usuário que não foi fornecido continuaria aqui)
    pass

## SUGESTÃO: LÓGICA DE VALIDAÇÃO FINAL
#
# No momento de submeter o formulário, antes de chamar `insert_registration`,
# adicione uma validação central para todos os campos.
#
# Exemplo:
#
# if st.button("Registrar"):
#     errors = []
#     if not st.session_state.get("first_name"):
#         errors.append("O campo 'Nome' é obrigatório.")
#     if not eid_is_valid(st.session_state.get("eid_live")):
#         errors.append("O EID fornecido é inválido.")
#     # ... adicione todas as outras validações aqui
#
#     if errors:
#         for error in errors:
#             st.error(error)
#     else:
#         # Se não houver erros, prossiga com a criação do registro
#         # ... coletar todos os dados do st.session_state ...
#         new_registration = { ... }
#         insert_registration(new_registration)
#         st.success("Inscrição realizada com sucesso!")
#         st.session_state["screen"] = "success" # Mudar para a tela de sucesso
