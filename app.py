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

# =========================
# CONFIG GERAL
# =========================
st.set_page_config(page_title="Inscrição – ADSS Jiu-Jitsu Competition", layout="centered")

# Secrets opcionais (apenas não quebre se não existir)
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
            id_doc_photo_path TEXT
        );
        """))
        # Tentar adicionar colunas novas se DB antigo existir (ignora erros)
        for alter in [
            "ALTER TABLE registrations ADD COLUMN first_name TEXT",
            "ALTER TABLE registrations ADD COLUMN last_name TEXT",
            "ALTER TABLE registrations ADD COLUMN full_name_en TEXT",
            "ALTER TABLE registrations ADD COLUMN coach_phone TEXT",
            "ALTER TABLE registrations ADD COLUMN age_years INTEGER"
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

# =========================
# UTIL: salvar imagem / arquivo
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
# NORMALIZAÇÃO / VALIDAÇÃO
# =========================
def squeeze_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def title_capitalize(s: str) -> str:
    # Coloca somente a primeira letra de cada palavra maiúscula
    return " ".join([w.capitalize() for w in squeeze_spaces(s).split(" ")])

def normalize_academy(s: str) -> str:
    return title_capitalize(s)

def clean_and_format_phone_05_mask(raw: str) -> str:
    # Expectativa: 05_-___-____  (ex.: 052-123-4567)
    digits = re.sub(r"\D", "", raw or "")
    if digits.startswith("9715"):  # comum no UAE com DDI
        digits = "05" + digits[-8:]
    if digits.startswith("5") and len(digits) == 9:
        digits = "0" + digits
    if len(digits) == 10 and digits.startswith("05"):
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    # fallback: retorna original limpo, mas deixa o usuário corrigir
    return raw

# Tradução para inglês (opcional, sem chave)
def translate_to_english(text_in: str) -> str:
    try:
        from deep_translator import GoogleTranslator
        t = GoogleTranslator(source="auto", target="en").translate(text_in)
        return title_capitalize(t)
    except Exception:
        # Se falhar, devolve normalizado em "title case"
        return title_capitalize(text_in)

# =========================
# LISTAS
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

# Países via pycountry
def get_country_list():
    try:
        import pycountry
        countries = [c.name for c in pycountry.countries]
        # Ajustes comuns
        extras = ["Hong Kong", "Macau", "Palestine", "Kosovo"]
        countries = sorted(set(countries + extras))
        return countries
    except Exception:
        # fallback simples
        return ["United Arab Emirates", "Brazil", "Portugal", "United States", "United Kingdom", "Italy", "Spain", "France", "Netherlands"]

# =========================
# CATEGORIA POR IDADE (baseada no ANO)
# =========================
def compute_age_year_based(year: int) -> int:
    if not year:
        return 0
    current_year = dt.date.today().year
    return max(0, current_year - year)

def age_division_by_year(age_year_based: int) -> str:
    # ajuste conforme seu regulamento. Aqui, regra simples:
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
# SIDEBAR (Evento / Admin / Branding)
# =========================
st.sidebar.header("Informações do Evento")
event_name = st.sidebar.text_input("Nome do evento", value="ADSS Jiu-Jitsu Competition")
region = st.sidebar.text_input("Região / Cidade", value="Al Dhafra – Abu Dhabi")

st.sidebar.markdown("---")
st.sidebar.subheader("Área do Organizador (Admin)")
admin_pw = st.sidebar.text_input("Senha admin", type="password", placeholder="••••••")
is_admin = (admin_pw == ADMIN_PASSWORD)

st.sidebar.markdown("---")
st.sidebar.subheader("Contato & Branding")
whatsapp_phone = st.sidebar.text_input("WhatsApp da organização (com DDI)", value="+9715xxxxxxxx")
logo_file = st.sidebar.file_uploader("Logo/Banner do evento (PNG/JPG)", type=["png", "jpg", "jpeg"])

# Botão flutuante WhatsApp
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
# MODO ADMIN
# =========================
def admin_view():
    st.title("Painel do Organizador")
    df = fetch_all()
    st.write(f"Total de inscrições: {len(df)}")

    if df.empty:
        st.info("Nenhuma inscrição encontrada.")
        return

    for _, row in df.iterrows():
        header = f"{row.get('first_name','')} {row.get('last_name','')} – {row.get('id','')}".strip()
        with st.expander(header or row.get('id','(sem id)')):
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
                "Criado em (UTC)": row.get("created_at","")
            }
            st.write(meta)
            if row.get("profile_photo_path") and Path(row["profile_photo_path"]).exists():
                st.image(row["profile_photo_path"], caption="Foto de Perfil", width=160)
            if row.get("id_doc_photo_path") and Path(row["id_doc_photo_path"]).exists():
                st.image(row["id_doc_photo_path"], caption="Documento", width=220)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Baixar CSV", data=csv, file_name="inscricoes.csv", mime="text/csv")

if is_admin:
    if logo_file is not None:
        st.image(logo_file, use_container_width=True)
    admin_view()
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

with st.form("registration_form", clear_on_submit=True):
    colA, colB = st.columns(2)
    with colA:
        first_name_raw = st.text_input("Nome*", max_chars=60, placeholder="Ex.: João")
        last_name_raw  = st.text_input("Sobrenome*", max_chars=60, placeholder="Ex.: Silva")
        email = st.text_input("E-mail*", placeholder="email@exemplo.com")
        phone_input = st.text_input("Telefone/WhatsApp* (formato 05_-___-____)", placeholder="052-123-4567")
        nationality = st.selectbox("Nacionalidade*", get_country_list())
        gender = st.selectbox("Gênero*", GENEROS, index=0)
    with colB:
        # Data de nascimento em 3 campos: dia/mes/ano
        st.write("Data de nascimento*")
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            dob_day = st.number_input("Dia", min_value=1, max_value=31, step=1, value=None, placeholder="DD")
        with c2:
            dob_month = st.number_input("Mês", min_value=1, max_value=12, step=1, value=None, placeholder="MM")
        with c3:
            current_year = dt.date.today().year
            dob_year = st.number_input("Ano", min_value=1900, max_value=current_year, step=1, value=None, placeholder="AAAA")

        academy_options = ["(Outro)"] + fetch_distinct_academies()
        academy_choice = st.selectbox("Academia*", academy_options)
        academy_other = ""
        if academy_choice == "(Outro)":
            academy_other = st.text_input("Escreva o nome da academia*", placeholder="Nome da academia")
        coach = st.text_input("Professor/Coach", placeholder="Nome do coach")
        coach_phone_input = st.text_input("Telefone do Professor (formato 05_-___-____)", placeholder="052-123-4567")

    # Fotos obrigatórias
    st.subheader("Fotos obrigatórias")
    profile_img = st.camera_input("Foto de Perfil (rosto visível)")
    id_doc_img = st.camera_input("Documento de Identificação (frente)")

    # Modalidade / Faixa / Peso
    st.subheader("Informações de Competição")
    modality = st.selectbox("Modalidade*", MODALIDADES, index=0)
    belt = st.selectbox("Faixa*", belts_for(modality))
    weight_class = st.selectbox("Peso*", weight_options(modality, gender))

    # Termo de consentimento
    st.subheader("Termo de Consentimento")
    st.write("Declaro estar apto(a) a participar e concordo com as regras do evento.")
    consent = st.checkbox("Eu li e concordo com o termo de consentimento*")

    notes = st.text_area("Observações (opcional)", height=80)

    submitted = st.form_submit_button("Enviar inscrição")

    if submitted:
        # Validações básicas
        missing = []
        if not first_name_raw: missing.append("Nome")
        if not last_name_raw: missing.append("Sobrenome")
        if not email: missing.append("E-mail")
        if not phone_input: missing.append("Telefone/WhatsApp")
        if dob_day is None or dob_month is None or dob_year is None: missing.append("Data de nascimento")
        if academy_choice == "(Outro)" and not academy_other: missing.append("Academia")
        if not profile_img: missing.append("Foto de Perfil")
        if not id_doc_img: missing.append("Documento de Identificação")
        if not consent: missing.append("Termo de consentimento")

        if missing:
            st.error("Por favor, preencha os campos obrigatórios: " + ", ".join(missing))
        else:
            # Normalizações
            first_name = title_capitalize(first_name_raw)
            last_name  = title_capitalize(last_name_raw)
            full_name_en = translate_to_english(f"{first_name} {last_name}")

            phone = clean_and_format_phone_05_mask(phone_input)
            coach_phone = clean_and_format_phone_05_mask(coach_phone_input) if coach_phone_input else ""

            academy = normalize_academy(academy_other if academy_choice == "(Outro)" else academy_choice)

            # Montar data de nascimento (validação simples)
            try:
                dob_date = dt.date(int(dob_year), int(dob_month), int(dob_day))
                dob_iso = dob_date.isoformat()
            except Exception:
                st.error("Data de nascimento inválida. Verifique dia/mês/ano.")
                st.stop()

            # Idade baseada no ano (pedido do cliente)
            age_years = compute_age_year_based(int(dob_year))
            age_div = age_division_by_year(age_years)

            # Categoria (faixa etária) = age_div
            category = age_div

            reg_id = str(uuid.uuid4())[:8].upper()
            profile_photo_path = save_image(profile_img, reg_id, "profile")
            id_doc_photo_path = save_image(id_doc_img, reg_id, "id_doc")

            row = {
                "id": reg_id,
                "created_at": dt.datetime.utcnow().isoformat(),
                "event_name": event_name,
                "first_name": first_name,
                "last_name": last_name,
                "full_name_en": full_name_en,
                "email": email.strip(),
                "phone": phone,
                "nationality": nationality,
                "gender": gender,
                "dob": dob_iso,
                "age_years": age_years,
                "age_division": age_div,
                "academy": academy,
                "coach": title_capitalize(coach),
                "coach_phone": coach_phone,
                "region": region,
                "modality": modality,
                "belt": belt,
                "weight_class": weight_class,
                "category": category,
                "consent": 1 if consent else 0,
                "notes": (notes or "").strip(),
                "profile_photo_path": profile_photo_path,
                "id_doc_photo_path": id_doc_photo_path,
            }

            try:
                insert_registration(row)
                st.success(f"Inscrição enviada com sucesso! Seu ID: {reg_id}")

                # Mostrar contagem de atletas naquela categoria
                total_cat = count_by_category(category)
                st.info(f"Atletas já inscritos na categoria '{category}': {total_cat}")

                # Link WhatsApp com ID
                if whatsapp_phone.strip():
                    msg = f"Olá! Minha inscrição do evento {event_name} foi enviada. Meu ID é {reg_id}."
                    phone_clean = whatsapp_phone.replace("+", "").replace(" ", "").replace("-", "")
                    wa_url = f"https://wa.me/{phone_clean}?text=" + urllib.parse.quote(msg)
                    st.markdown(f"[Falar com a organização no WhatsApp]({wa_url})")
            except Exception as e:
                st.error(f"Erro ao salvar inscrição: {e}")
