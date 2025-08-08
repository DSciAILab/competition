import os
import re
import uuid
import urllib.parse
import datetime as dt
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from PIL import Image

# Face detection
import cv2
import numpy as np

# =========================
# CONFIG GERAL
# =========================
st.set_page_config(page_title="Inscrição – ADSS Jiu-Jitsu Competition", layout="centered")

# Secrets opcionais (não falhar se não existir)
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", None)
try:
    if "ADMIN_PASSWORD" in st.secrets:
        ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
except Exception:
    pass
if not ADMIN_PASSWORD:
    ADMIN_PASSWORD = "admin123"

DB_PATH = "registrations.db"
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# =========================
# BANCO DE DADOS (SQLite)
# =========================
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
            category TEXT,
            consent INTEGER,
            notes TEXT,
            profile_photo_path TEXT,
            id_doc_photo_path TEXT,
            approval_status TEXT
        );
        """))
        # Migrações leves (ignora se já existem)
        for alter in [
            "ALTER TABLE registrations ADD COLUMN first_name TEXT",
            "ALTER TABLE registrations ADD COLUMN last_name TEXT",
            "ALTER TABLE registrations ADD COLUMN full_name_en TEXT",
            "ALTER TABLE registrations ADD COLUMN coach_phone TEXT",
            "ALTER TABLE registrations ADD COLUMN age_years INTEGER",
            "ALTER TABLE registrations ADD COLUMN approval_status TEXT"
        ]:
            try:
                conn.execute(text(alter))
            except Exception:
                pass

init_db()

def insert_registration(row: dict):
    with engine.begin() as conn:
        cols = ",".join(row.keys())
        vals = ",".join([f":{k}" for k in row.keys()])
        conn.execute(text(f"INSERT INTO registrations ({cols}) VALUES ({vals})"), row)

def fetch_all():
    with engine.begin() as conn:
        df = pd.read_sql("SELECT * FROM registrations ORDER BY created_at DESC", conn)
    return df

def fetch_distinct_academies():
    with engine.begin() as conn:
        try:
            rows = conn.execute(text(
                "SELECT DISTINCT academy FROM registrations WHERE academy IS NOT NULL AND academy <> ''"
            )).fetchall()
            return sorted({r[0] for r in rows})
        except Exception:
            return []

def count_by_category(category_value: str):
    with engine.begin() as conn:
        try:
            row = conn.execute(text(
                "SELECT COUNT(*) FROM registrations WHERE category = :c"
            ), {"c": category_value}).fetchone()
            return int(row[0] if row else 0)
        except Exception:
            return 0

# =========================
# UTIL: salvar imagem
# =========================
def save_image(file, reg_id: str, suffix: str, max_size=(800, 800)) -> str:
    if file is None:
        return ""
    image = Image.open(file)
    image.thumbnail(max_size)
    dest = UPLOAD_DIR / f"{reg_id}_{suffix}.jpg"
    image.save(dest, format="JPEG", quality=85)
    return str(dest)

# =========================
# FACE DETECTION
# =========================
def count_faces_in_image(file) -> int:
    """Retorna o número de faces detectadas na imagem (UploadedFile/camera_input)."""
    if file is None:
        return 0
    bytes_data = file.getvalue()
    np_arr = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    return 0 if faces is None else len(faces)

# =========================
# NORMALIZAÇÃO / VALIDAÇÃO
# =========================
def squeeze_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def title_capitalize(s: str) -> str:
    # Somente a primeira letra de cada palavra maiúscula
    return " ".join([w.capitalize() for w in squeeze_spaces(s).split(" ")])

def normalize_academy(s: str) -> str:
    return title_capitalize(s)

def mask_and_validate_phone_live(key: str):
    """Formata ao digitar para 05_-___-____ e limita a 10 dígitos."""
    raw = st.session_state.get(key, "") or ""
    digits = re.sub(r"\D", "", raw)
    if digits.startswith("9715"):
        digits = "05" + digits[-8:]
    if digits.startswith("5") and len(digits) <= 9:
        digits = "0" + digits
    digits = digits[:10]
    if len(digits) <= 3:
        out = digits
    elif len(digits) <= 6:
        out = f"{digits[:3]}-{digits[3:]}"
    else:
        out = f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    st.session_state[key] = out

def clean_and_format_phone_final(raw: str) -> str:
    """Valida e normaliza no submit (mantém 052-123-4567)."""
    digits = re.sub(r"\D", "", raw or "")
    if digits.startswith("9715"):
        digits = "05" + digits[-8:]
    if digits.startswith("5") and len(digits) == 9:
        digits = "0" + digits
    digits = digits[:10]
    if len(digits) == 10 and digits.startswith("05"):
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    return raw

def translate_to_english(text_in: str) -> str:
    try:
        from deep_translator import GoogleTranslator
        t = GoogleTranslator(source="auto", target="en").translate(text_in)
        return title_capitalize(t)
    except Exception:
        return title_capitalize(text_in)

# =========================
# LISTAS / OPÇÕES
# =========================
MODALIDADES = ["Gi", "No-Gi"]
GENEROS = ["Masculino", "Feminino"]

FAIXAS_GI = ["Branca", "Cinza (Kids)", "Amarela (Kids)", "Laranja (Kids)", "Verde (Kids)", "Azul", "Roxa", "Marrom", "Preta"]
FAIXAS_NOGI = ["Beginner", "Intermediate", "Advanced"]

PESOS_ADULT_MASC_GI = ["-56.5kg", "-62kg", "-69kg", "-77kg", "-85.5kg", "-94kg", "-102kg", "+102kg"]
PESOS_ADULT_FEM_GI  = ["-48.5kg", "-53.5kg", "-58.5kg", "-64kg", "-69kg", "-74kg", "+74kg"]
PESOS_NOGI = ["-60kg", "-66kg", "-73kg", "-80kg", "-87kg", "-94kg", "-102kg", "+102kg"]

def weight_options(modalidade: str, genero: str):
    if modalidade == "Gi":
        return PESOS_ADULT_MASC_GI if genero == "Masculino" else PESOS_ADULT_FEM_GI
    return PESOS_NOGI

def get_country_list():
    try:
        import pycountry
        countries = [c.name for c in pycountry.countries]
        extras = ["Hong Kong", "Macau", "Palestine", "Kosovo"]
        countries = sorted(set(countries + extras))
        return countries
    except Exception:
        return ["United Arab Emirates", "Brazil", "Portugal", "United States", "United Kingdom", "Italy", "Spain", "France", "Netherlands"]

# =========================
# CATEGORIA POR IDADE (base ANO)
# =========================
def compute_age_year_based(year: int) -> int:
    if not year:
        return 0
    current_year = dt.date.today().year
    return max(0, current_year - year)

def age_division_by_year(age_year_based: int) -> str:
    if age_year_based <= 15:
        return "Kids"
    if 16 <= age_year_based <= 17:
        return "Juvenile"
    if 18 <= age_year_based <= 29:
        return "Adult"
    if 30 <= age_year_based <= 35:
        return "Master 1"
    if 36 <= age_year_based <= 40:
        return "Master 2"
    if 41 <= age_year_based <= 45:
        return "Master 3"
    return "Master 4+"

def belts_for(modalidade: str):
    return FAIXAS_GI if modalidade == "Gi" else FAIXAS_NOGI

# =========================
# SIDEBAR (Evento / Admin / Branding / Navegação)
# =========================
st.sidebar.header("Informações do Evento")
event_name = st.sidebar.text_input("Nome do evento", value="ADSS Jiu-Jitsu Competition")
region = st.sidebar.text_input("Região / Cidade", value="Al Dhafra – Abu Dhabi")

st.sidebar.markdown("---")
st.sidebar.subheader("Área do Organizador (Admin)")
admin_pw = st.sidebar.text_input("Senha admin", type="password", placeholder="••••••")
is_admin = (admin_pw == ADMIN_PASSWORD)

st.sidebar.markdown("---")
st.sidebar.subheader("Contato e Branding")
whatsapp_phone = st.sidebar.text_input("WhatsApp da organização (com DDI)", value="+9715xxxxxxxx")
logo_file = st.sidebar.file_uploader("Logo/Banner do evento (PNG/JPG)", type=["png", "jpg", "jpeg"])

st.sidebar.markdown("---")
page = st.sidebar.radio("Página", ["Formulário", "Estatísticas"], index=0)

# =========================
# Botão flutuante WhatsApp
# =========================
def whatsapp_button(phone: str, event_name: str):
    if not phone.strip():
        return
    phone_clean = phone.replace("+", "").replace(" ", "").replace("-", "")
    default_msg = f"Olá! Tenho uma dúvida sobre o evento {event_name}."
    url = f"https://wa.me/{phone_clean}?text=" + urllib.parse.quote(default_msg)
    st.markdown(
        f"""
        <a href="{url}" target="_blank"
           style="
             position: fixed;
             right: 20px;
             bottom: 20px;
             background: #25D366;
             color: #fff;
             padding: 10px 14px;
             border-radius: 999px;
             text-decoration: none;
             font-weight: 600;
             box-shadow: 0 4px 16px rgba(0,0,0,.2);
             z-index: 9999;
             font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
           ">
           WhatsApp da Organização
        </a>
        """,
        unsafe_allow_html=True
    )
whatsapp_button(whatsapp_phone, event_name)

# =========================
# ESTADO: erros por campo (para highlight)
# =========================
if "errors" not in st.session_state:
    st.session_state["errors"] = set()

def label(label_text: str, key: str, required: bool = False):
    err = key in st.session_state["errors"]
    color = "#b91c1c" if err else "#111"
    star = " *" if required else ""
    st.markdown(f"<div style='font-weight:600;color:{color}'>{label_text}{star}</div>", unsafe_allow_html=True)

# =========================
# VIEWS
# =========================
def admin_view():
    st.title("Painel do Organizador")
    df = fetch_all()
    st.write(f"Total de inscrições: {len(df)}")

    if df.empty:
        st.info("Ainda não há inscrições.")
        return

    st.markdown("""
        <style>
        .badge-pending {
            display:inline-block;
            padding:2px 8px;
            border-radius:999px;
            background:#F59E0B;
            color:#111;
            font-size:12px;
            font-weight:600;
            margin-left:8px;
        }
        </style>
    """, unsafe_allow_html=True)

    for _, row in df.iterrows():
        name = f"{row.get('first_name','')} {row.get('last_name','')}".strip()
        status = (row.get("approval_status") or "Pending").strip()
        badge_html = "<span class='badge-pending'>Pendente de aprovação</span>" if status.lower()=="pending" else ""
        header_html = (name or row.get('id','(sem id)')) + " " + badge_html

        with st.expander(header_html, expanded=False):
            meta = {
                "Nome (EN)": row.get("full_name_en",""),
                "E-mail": row.get("email",""),
                "Telefone": row.get("phone",""),
                "Nacionalidade": row.get("nationality",""),
                "Gênero": row.get("gender",""),
                "Data de nascimento": row.get("dob",""),
                "Idade (ano-base)": row.get("age_years",""),
                "Divisão etária": row.get("age_division",""),
                "Academia": row.get("academy",""),
                "Professor": row.get("coach",""),
                "Telefone do professor": row.get("coach_phone",""),
                "Região": row.get("region",""),
                "Modalidade": row.get("modality",""),
                "Faixa": row.get("belt",""),
                "Peso": row.get("weight_class",""),
                "Categoria": row.get("category",""),
                "Status": status,
                "Criado em (UTC)": row.get("created_at","")
            }
            st.write(meta)
            if row.get("profile_photo_path") and Path(row["profile_photo_path"]).exists():
                st.image(row["profile_photo_path"], caption="Foto de Perfil", width=160)
            if row.get("id_doc_photo_path") and Path(row["id_doc_photo_path"]).exists():
                st.image(row["id_doc_photo_path"], caption="Documento", width=220)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Baixar CSV", data=csv, file_name="inscricoes.csv", mime="text/csv")

def stats_view():
    st.title("Estatísticas – Inscritos por Categoria")
    df = fetch_all()
    if df.empty:
        st.info("Ainda não há inscrições.")
        return
    if "category" not in df.columns:
        st.error("Não foi encontrada a coluna 'category' na base.")
        return

    totals = (
        df.groupby("category", dropna=False)
          .size()
          .reset_index(name="inscritos")
          .sort_values("inscritos", ascending=False)
          .reset_index(drop=True)
    )
    st.subheader("Total por categoria")
    st.dataframe(totals, use_container_width=True)
    st.bar_chart(totals.set_index("category"))

    st.subheader("Detalhamento por modalidade dentro de cada categoria")
    if {"category", "modality"}.issubset(df.columns):
        pivot_mod = df.pivot_table(index="category", columns="modality", values="id", aggfunc="count", fill_value=0)
        st.dataframe(pivot_mod, use_container_width=True)

    st.subheader("Detalhamento por faixa dentro de cada categoria")
    if {"category", "belt"}.issubset(df.columns):
        pivot_belt = df.pivot_table(index="category", columns="belt", values="id", aggfunc="count", fill_value=0)
        st.dataframe(pivot_belt, use_container_width=True)

# =========================
# ROTEAMENTO
# =========================
if is_admin:
    if logo_file is not None:
        st.image(logo_file, use_container_width=True)
    admin_view()
    st.stop()

if page == "Estatísticas":
    if logo_file is not None:
        st.image(logo_file, use_container_width=True)
    stats_view()
    st.stop()

# =========================
# CABEÇALHO
# =========================
if logo_file is not None:
    st.image(logo_file, use_container_width=True)
else:
    st.markdown(f"<h1 style='text-align:center;margin-bottom:0'>{event_name}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;margin-top:4px;color:#777'>{region}</p>", unsafe_allow_html=True)

# =========================
# FORMULÁRIO
# =========================
st.title("Formulário de Inscrição")
st.caption("Preencha seus dados. Campos com * são obrigatórios.")

with st.form("registration_form", clear_on_submit=False):
    # ------ Coluna A ------
    colA, colB = st.columns(2)
    with colA:
        label("Nome", "first_name", required=True)
        first_name_raw = st.text_input("", key="first_name", label_visibility="collapsed", max_chars=60, placeholder="Ex.: João")

        label("Sobrenome", "last_name", required=True)
        last_name_raw  = st.text_input("", key="last_name", label_visibility="collapsed", max_chars=60, placeholder="Ex.: Silva")

        label("E-mail", "email", required=True)
        email = st.text_input("", key="email", label_visibility="collapsed", placeholder="email@exemplo.com")

        label("Telefone/WhatsApp (formato 05_-___-____)", "phone", required=True)
        phone_input = st.text_input(
            "", key="phone", label_visibility="collapsed",
            placeholder="052-123-4567", on_change=mask_and_validate_phone_live, args=("phone",)
        )

        label("Nacionalidade", "nationality", required=True)
        nationality = st.selectbox("", get_country_list(), key="nationality", label_visibility="collapsed")

        label("Gênero", "gender", required=True)
        gender = st.selectbox("", ["Masculino","Feminino"], key="gender", label_visibility="collapsed")

    # ------ Coluna B ------
    with colB:
        st.write("Data de nascimento")
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            label("Dia", "dob_day", required=True)
            dob_day = st.number_input("", key="dob_day", label_visibility="collapsed", min_value=1, max_value=31, step=1, value=st.session_state.get("dob_day", None))
        with c2:
            label("Mês", "dob_month", required=True)
            dob_month = st.number_input("", key="dob_month", label_visibility="collapsed", min_value=1, max_value=12, step=1, value=st.session_state.get("dob_month", None))
        with c3:
            current_year = dt.date.today().year
            label("Ano", "dob_year", required=True)
            dob_year = st.number_input("", key="dob_year", label_visibility="collapsed", min_value=1900, max_value=current_year, step=1, value=st.session_state.get("dob_year", None))

        academies = fetch_distinct_academies()
        options = ["Selecione...", "Minha academia não está na lista"] + academies
        label("Academia", "academy_choice", required=True)
        academy_choice = st.selectbox("", options, key="academy_choice", label_visibility="collapsed", index=st.session_state.get("academy_choice_idx", 0))
        academy_other = ""
        if academy_choice == "Minha academia não está na lista":
            label("Nome da academia", "academy_other", required=True)
            academy_other = st.text_input("", key="academy_other", label_visibility="collapsed", placeholder="Nome da academia")

        label("Professor/Coach", "coach", required=False)
        coach = st.text_input("", key="coach", label_visibility="collapsed", placeholder="Nome do coach")

        label("Telefone do Professor (formato 05_-___-____)", "coach_phone", required=False)
        coach_phone_input = st.text_input(
            "", key="coach_phone", label_visibility="collapsed",
            placeholder="052-123-4567", on_change=mask_and_validate_phone_live, args=("coach_phone",)
        )

    st.subheader("Fotos obrigatórias")
    label("Foto de Perfil (rosto visível)", "profile_img", required=True)
    profile_img = st.camera_input("", key="profile_img", label_visibility="collapsed")

    label("Documento de Identificação (frente)", "id_doc_img", required=True)
    id_doc_img = st.camera_input("", key="id_doc_img", label_visibility="collapsed")

    st.subheader("Informações de Competição")
    label("Modalidade", "modality", required=True)
    modality = st.selectbox("", MODALIDADES, key="modality", label_visibility="collapsed")

    label("Faixa", "belt", required=True)
    belt = st.selectbox("", belts_for(modality), key="belt", label_visibility="collapsed")

    label("Peso", "weight_class", required=True)
    weight_class = st.selectbox("", weight_options(modality, gender), key="weight_class", label_visibility="collapsed")

    st.subheader("Termo de Consentimento")
    st.write("Declaro estar apto(a) a participar e concordo com as regras do evento.")
    consent = st.checkbox("Eu li e concordo com o termo de consentimento*", key="consent")

    notes = st.text_area("Observações (opcional)", height=80, key="notes")

    # Botão de envio DENTRO do form
    submitted = st.form_submit_button("Enviar inscrição")

# =========================
# PROCESSAMENTO DO SUBMIT
# =========================
if submitted:
    # Verifica obrigatórios
    errors = set()
    if not st.session_state.get("first_name"): errors.add("first_name")
    if not st.session_state.get("last_name"):  errors.add("last_name")
    if not st.session_state.get("email"):      errors.add("email")
    if not st.session_state.get("phone"):      errors.add("phone")
    if not st.session_state.get("gender"):     errors.add("gender")
    if not st.session_state.get("nationality"):errors.add("nationality")
    if not st.session_state.get("modality"):   errors.add("modality")
    if not st.session_state.get("belt"):       errors.add("belt")
    if not st.session_state.get("weight_class"): errors.add("weight_class")
    if st.session_state.get("academy_choice") in (None, "", "Selecione..."): errors.add("academy_choice")
    if st.session_state.get("academy_choice") == "Minha academia não está na lista" and not st.session_state.get("academy_other"):
        errors.add("academy_other")
    if st.session_state.get("dob_day") in (None, ""):   errors.add("dob_day")
    if st.session_state.get("dob_month") in (None, ""): errors.add("dob_month")
    if st.session_state.get("dob_year") in (None, ""):  errors.add("dob_year")
    if st.session_state.get("profile_img") is None:     errors.add("profile_img")
    if st.session_state.get("id_doc_img") is None:      errors.add("id_doc_img")
    if not st.session_state.get("consent"):             errors.add("consent")

    # Validação de rosto na foto de perfil
    if "profile_img" not in errors and st.session_state.get("profile_img") is not None:
        faces_profile = count_faces_in_image(st.session_state.get("profile_img"))
        if faces_profile == 0:
            errors.add("profile_img")
            st.error("A foto de perfil não parece conter um rosto. Por favor, envie uma foto clara do rosto.")
        elif faces_profile > 1:
            errors.add("profile_img")
            st.error("Detectamos mais de uma pessoa na foto de perfil. Envie uma foto só do atleta.")

    st.session_state["errors"] = errors  # para destacar em vermelho

    if errors:
        st.error("Há campos obrigatórios faltando. Eles estão destacados em vermelho acima.")
        st.stop()

    # Normalizações
    first_name = title_capitalize(st.session_state["first_name"])
    last_name  = title_capitalize(st.session_state["last_name"])
    full_name_en = translate_to_english(f"{first_name} {last_name}")

    phone = clean_and_format_phone_final(st.session_state["phone"])
    coach_phone = clean_and_format_phone_final(st.session_state.get("coach_phone","")) if st.session_state.get("coach_phone") else ""

    if st.session_state["academy_choice"] == "Minha academia não está na lista":
        academy = normalize_academy(st.session_state["academy_other"])
    else:
        academy = normalize_academy(st.session_state["academy_choice"])

    # Data
    try:
        dob_date = dt.date(int(st.session_state["dob_year"]), int(st.session_state["dob_month"]), int(st.session_state["dob_day"]))
        dob_iso = dob_date.isoformat()
    except Exception:
        st.error("Data de nascimento inválida. Verifique dia, mês e ano.")
        st.stop()

    # Idade por ano-base (categoria pelo ANO)
    age_years = compute_age_year_based(int(st.session_state["dob_year"]))
    age_div = age_division_by_year(age_years)
    category = age_div

    reg_id = str(uuid.uuid4())[:8].upper()
    profile_photo_path = save_image(st.session_state["profile_img"], reg_id, "profile")
    id_doc_photo_path = save_image(st.session_state["id_doc_img"], reg_id, "id_doc")

    row = {
        "id": reg_id,
        "created_at": dt.datetime.utcnow().isoformat(),
        "event_name": event_name,
        "first_name": first_name,
        "last_name": last_name,
        "full_name_en": full_name_en,
        "email": st.session_state["email"].strip(),
        "phone": phone,
        "nationality": st.session_state["nationality"],
        "gender": st.session_state["gender"],
        "dob": dob_iso,
        "age_years": age_years,
        "age_division": age_div,
        "academy": academy,
        "coach": title_capitalize(st.session_state.get("coach","")),
        "coach_phone": coach_phone,
        "region": region,
        "modality": st.session_state["modality"],
        "belt": st.session_state["belt"],
        "weight_class": st.session_state["weight_class"],
        "category": category,
        "consent": 1,
        "notes": (st.session_state.get("notes") or "").strip(),
        "profile_photo_path": profile_photo_path,
        "id_doc_photo_path": id_doc_photo_path,
        "approval_status": "Pending"
    }

    try:
        insert_registration(row)
        st.success(f"Inscrição enviada com sucesso. ID: {reg_id}")
        st.info("Status da sua inscrição: Pendente de aprovação.")

        total_cat = count_by_category(category)
        st.info(f"Atletas já inscritos na categoria '{category}': {total_cat}")

        if whatsapp_phone.strip():
            msg = f"Olá! Minha inscrição do evento {event_name} foi enviada. Meu ID é {reg_id}."
            phone_clean = whatsapp_phone.replace("+", "").replace(" ", "").replace("-", "")
            wa_url = f"https://wa.me/{phone_clean}?text=" + urllib.parse.quote(msg)
            st.markdown(f"[Falar com a organização no WhatsApp]({wa_url})")
    except Exception as e:
        st.error(f"Erro ao salvar inscrição: {e}")
