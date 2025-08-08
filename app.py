import os
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
st.set_page_config(page_title="Inscrição – Campeonato de Jiu-Jítsu", page_icon="🥋", layout="centered")

# Carrega secrets (opcional) para env vars
if "ADMIN_PASSWORD" in st.secrets:
    os.environ["ADMIN_PASSWORD"] = st.secrets["ADMIN_PASSWORD"]

DB_PATH = "registrations.db"
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")  # troque em produção

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
            full_name TEXT,
            email TEXT,
            phone TEXT,
            nationality TEXT,
            gender TEXT,
            dob TEXT,
            age_division TEXT,
            academy TEXT,
            coach TEXT,
            region TEXT,
            modality TEXT,
            belt TEXT,
            weight_class TEXT,
            category TEXT,
            emergency_contact TEXT,
            medical_notes TEXT,
            consent INTEGER,
            payment_status TEXT,
            payment_note TEXT,
            proof_path TEXT,
            notes TEXT,
            profile_photo_path TEXT,
            id_doc_photo_path TEXT
        );
        """))

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

# =========================
# ARQUIVOS: salvar imagem (com compressão) e arquivo genérico
# =========================
def save_image(file, reg_id: str, suffix: str, max_size=(800, 800)) -> str:
    """
    Salva imagens (camera_input ou upload de imagem) comprimindo para JPG.
    """
    if file is None:
        return ""
    image = Image.open(file)
    image.thumbnail(max_size)  # reduz mantendo proporção
    dest = UPLOAD_DIR / f"{reg_id}_{suffix}.jpg"
    image.save(dest, format="JPEG", quality=85)
    return str(dest)

def save_file_generic(file, reg_id: str, suffix: str) -> str:
    """
    Salva qualquer arquivo (PDF/IMG). Mantém a extensão original quando houver.
    """
    if file is None:
        return ""
    ext = Path(file.name).suffix.lower() or ".bin"
    dest = UPLOAD_DIR / f"{reg_id}_{suffix}{ext}"
    with open(dest, "wb") as f:
        f.write(file.getbuffer())
    return str(dest)

# =========================
# LISTAS DE REFERÊNCIA
# =========================
MODALIDADES = ["Gi", "No-Gi"]
GENEROS = ["Masculino", "Feminino"]
CATEGORIAS = ["Kids", "Juvenile", "Adult", "Master 1", "Master 2", "Master 3", "Master 4+"]

FAIXAS_GI = [
    "Branca", "Cinza (Kids)", "Amarela (Kids)", "Laranja (Kids)", "Verde (Kids)",
    "Azul", "Roxa", "Marrom", "Preta"
]
FAIXAS_NOGI = ["Beginner", "Intermediate", "Advanced"]

PESOS_ADULT_MASC_GI = ["-56.5kg", "-62kg", "-69kg", "-77kg", "-85.5kg", "-94kg", "-102kg", "+102kg"]
PESOS_ADULT_FEM_GI  = ["-48.5kg", "-53.5kg", "-58.5kg", "-64kg", "-69kg", "-74kg", "+74kg"]
PESOS_NOGI = ["-60kg", "-66kg", "-73kg", "-80kg", "-87kg", "-94kg", "-102kg", "+102kg"]

def weight_options(modalidade: str, genero: str, categoria: str):
    if modalidade == "Gi":
        return PESOS_ADULT_MASC_GI if genero == "Masculino" else PESOS_ADULT_FEM_GI
    return PESOS_NOGI

def belts_for(modalidade: str, categoria: str):
    return FAIXAS_GI if modalidade == "Gi" else FAIXAS_NOGI

def infer_age_division(dob: dt.date) -> str:
    if not dob:
        return "Adult"
    today = dt.date.today()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    if age <= 15:
        return "Kids"
    if 16 <= age <= 17:
        return "Juvenile"
    if 18 <= age <= 29:
        return "Adult"
    if 30 <= age <= 35:
        return "Master 1"
    if 36 <= age <= 40:
        return "Master 2"
    if 41 <= age <= 45:
        return "Master 3"
    return "Master 4+"

# =========================
# SIDEBAR (Evento / Admin / Branding)
# =========================
st.sidebar.header("Informações do Evento")
event_name = st.sidebar.text_input("Nome do evento", value="Al Dhafra Open 2025")
region = st.sidebar.text_input("Região / Cidade", value="Al Dhafra – Abu Dhabi")

st.sidebar.markdown("---")
st.sidebar.subheader("Área do Organizador (Admin)")
admin_pw_input = st.sidebar.text_input("Senha admin", type="password", placeholder="••••••")
is_admin = (admin_pw_input == ADMIN_PASSWORD)

st.sidebar.markdown("---")
st.sidebar.subheader("Contato & Branding")
whatsapp_phone = st.sidebar.text_input("WhatsApp da organização (com DDI)", value="+9715xxxxxxxx")
logo_file = st.sidebar.file_uploader("Logo/Banner do evento (PNG/JPG)", type=["png", "jpg", "jpeg"])

# =========================
# BOTÃO FLUTUANTE WHATSAPP
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
             padding: 12px 16px;
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

# Render do botão flutuante
whatsapp_button(whatsapp_phone, event_name)

# =========================
# MODO ADMIN
# =========================
def admin_view():
    st.title("📊 Painel do Organizador")
    df = fetch_all()
    st.write(f"Total de inscrições: **{len(df)}**")

    if df.empty:
        st.info("Nenhuma inscrição encontrada.")
        return

    for _, row in df.iterrows():
        header = f"{row.get('full_name','(sem nome)')} – {row.get('id','')}"
        with st.expander(header):
            meta_cols = [
                "event_name","email","phone","nationality","gender","dob","age_division",
                "academy","coach","region","modality","belt","weight_class","category",
                "payment_status","payment_note","created_at"
            ]
            clean = {k: row[k] for k in meta_cols if k in row}
            st.write(clean)

            if row.get("profile_photo_path") and Path(row["profile_photo_path"]).exists():
                st.image(row["profile_photo_path"], caption="Foto de Perfil", width=160)
            if row.get("id_doc_photo_path") and Path(row["id_doc_photo_path"]).exists():
                st.image(row["id_doc_photo_path"], caption="Documento", width=220)
            if row.get("proof_path") and Path(row["proof_path"]).exists():
                st.write("📎 Comprovante de pagamento:")
                if Path(row["proof_path"]).suffix.lower() in [".jpg",".jpeg",".png"]:
                    st.image(row["proof_path"], caption="Comprovante", width=220)
                else:
                    st.code(row["proof_path"])

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Baixar CSV", data=csv, file_name="inscricoes.csv", mime="text/csv")

if is_admin:
    if logo_file is not None:
        st.image(logo_file, use_container_width=True)
    admin_view()
    st.stop()

# =========================
# CABEÇALHO (Logo/Texto)
# =========================
if logo_file is not None:
    st.image(logo_file, use_container_width=True)
else:
    st.markdown(f"<h1 style='text-align:center;margin-bottom:0'>{event_name}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;margin-top:4px;color:#777'>{region}</p>", unsafe_allow_html=True)

# =========================
# FORMULÁRIO DE INSCRIÇÃO
# =========================
st.title("🥋 Formulário de Inscrição")
st.caption("Preencha seus dados. Campos com * são obrigatórios.")

with st.form("registration_form", clear_on_submit=True):
    colA, colB = st.columns(2)
    with colA:
        full_name = st.text_input("Nome completo*", max_chars=120)
        email = st.text_input("E-mail*", placeholder="seu@email.com")
        phone = st.text_input("Telefone/WhatsApp*", placeholder="+971...")
        nationality = st.text_input("Nacionalidade", placeholder="Brazilian")
        gender = st.selectbox("Gênero*", GENEROS, index=0)
        dob = st.date_input("Data de nascimento*", value=None, format="DD/MM/YYYY")
    with colB:
        academy = st.text_input("Academia*", placeholder="Nome da academia")
        coach = st.text_input("Professor/Coach", placeholder="Nome do coach")
        modality = st.selectbox("Modalidade*", MODALIDADES, index=0)
        category = st.selectbox("Categoria (faixa etária)*", CATEGORIAS, index=2)
        belt = st.selectbox("Faixa*", belts_for(modality, category))
        weight_class = st.selectbox("Peso*", weight_options(modality, gender, category))

    st.markdown("### 📸 Fotos obrigatórias")
    st.subheader("1) Foto de Perfil")
    profile_img = st.camera_input(
        "Tire agora a sua foto de perfil",
        help="Rosto visível, sem boné ou óculos escuros"
    )

    st.subheader("2) Documento de Identificação")
    id_doc_img = st.camera_input(
        "Tire uma foto do documento (frente)",
        help="Foto legível, bem iluminada"
    )

    st.markdown("### Informações Médicas e Contato")
    emergency_contact = st.text_input("Contato de Emergência*", placeholder="Nome + telefone")
    medical_notes = st.text_area("Observações médicas (opcional)", height=80)

    st.markdown("### Pagamento")
    payment_status = st.selectbox("Status do pagamento*", ["Pendente", "Pago", "Isento"], index=0)
    payment_note = st.text_input("Observação do pagamento (opcional)")
    proof = st.file_uploader("Comprovante (PDF/Imagem)", type=["pdf", "png", "jpg", "jpeg"])

    st.markdown("### Termo de Consentimento")
    st.write("Declaro estar apto(a) a participar e concordo com as regras do evento.")
    consent = st.checkbox("Eu li e concordo com o termo de consentimento*")

    notes = st.text_area("Observações adicionais (opcional)", height=80)

    submitted = st.form_submit_button("Enviar inscrição")

    if submitted:
        required = [full_name, email, phone, academy, gender, belt, weight_class, emergency_contact, profile_img, id_doc_img]
        if (not consent) or any(x is None or (isinstance(x, str) and x.strip() == "") for x in required) or (dob is None):
            st.error("Preencha todos os campos obrigatórios e envie as duas fotos.")
        else:
            reg_id = str(uuid.uuid4())[:8].upper()

            proof_path = ""
            if proof is not None:
                if Path(proof.name).suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    proof_path = save_image(proof, reg_id, "payment_proof")
                else:
                    proof_path = save_file_generic(proof, reg_id, "payment_proof")

            profile_photo_path = save_image(profile_img, reg_id, "profile")
            id_doc_photo_path = save_image(id_doc_img, reg_id, "id_doc")

            inferred_division = infer_age_division(dob)

            row = {
                "id": reg_id,
                "created_at": dt.datetime.utcnow().isoformat(),
                "event_name": event_name,
                "full_name": full_name.strip(),
                "email": email.strip(),
                "phone": phone.strip(),
                "nationality": nationality.strip(),
                "gender": gender,
                "dob": dob.isoformat(),
                "age_division": category if category else inferred_division,
                "academy": academy.strip(),
                "coach": coach.strip(),
                "region": region.strip(),
                "modality": modality,
                "belt": belt,
                "weight_class": weight_class,
                "category": category,
                "emergency_contact": emergency_contact.strip(),
                "medical_notes": medical_notes.strip(),
                "consent": 1 if consent else 0,
                "payment_status": payment_status,
                "payment_note": payment_note.strip(),
                "proof_path": proof_path,
                "notes": notes.strip(),
                "profile_photo_path": profile_photo_path,
                "id_doc_photo_path": id_doc_photo_path,
            }

            try:
                insert_registration(row)
                st.success(f"Inscrição enviada com sucesso! Seu ID: **{reg_id}**")

                if whatsapp_phone.strip():
                    msg = f"Olá! Minha inscrição do evento {event_name} foi enviada. Meu ID é {reg_id}."
                    phone_clean = whatsapp_phone.replace("+", "").replace(" ", "").replace("-", "")
                    wa_url = f"https://wa.me/{phone_clean}?text=" + urllib.parse.quote(msg)
                    st.markdown(f"[💬 Falar com a organização no WhatsApp]({wa_url})")

            except Exception as e:
                st.error(f"Erro ao salvar inscrição: {e}")
