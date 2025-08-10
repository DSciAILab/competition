# ================================
# app.py – ADSS Jiu-Jitsu Competition
# Estruturado e comentado (mobile-first, full screen)
# ================================

# ---------- IMPORTS PADRÃO ----------
import os                               # Acesso a variáveis de ambiente e caminhos
import re                               # Regex para normalização de strings
import uuid                             # Geração de IDs únicos para inscrição
import urllib.parse                     # Montar URLs (WhatsApp)
import datetime as dt                   # Datas e horários
from pathlib import Path                # Manipulação de caminhos multiplataforma
from io import BytesIO                  # Buffer em memória para imagens/bytes

# ---------- IMPORTS DE TERCEIROS ----------
import pandas as pd                     # Tabelas/consultas simples
import streamlit as st                  # UI/Webapp
from sqlalchemy import create_engine, text  # Banco de dados SQLite via SQLAlchemy
from PIL import Image                   # Processamento básico de imagens (Pillow)

# ---------- VISÃO COMPUTACIONAL / HASH ----------
import numpy as np                      # Arrays numéricos (base para OpenCV/PIL)
try:
    import cv2                          # OpenCV (pode falhar em ambientes sem GUI)
    FACE_DETECT_AVAILABLE = True        # Flag indica se temos OpenCV
except Exception:                       # Se não conseguir importar
    cv2 = None                          # Mantém referência nula
    FACE_DETECT_AVAILABLE = False       # Marca como indisponível (tolerante)

import imagehash                        # pHash perceptual para comparação de imagens

# ================================
# 1) CONFIG GLOBAL (FULL SCREEN + MOBILE)
# ================================

st.set_page_config(                     # Define título e layout base da página
    page_title="ADSS Jiu-Jitsu Competition",
    layout="wide"                       # "wide" ajuda no aproveitamento de tela
)

# CSS para visual mobile-first e full screen (esconde header/menu/footer do Streamlit)
st.markdown("""
<style>
/* Esconde cabeçalho/menu/rodapé do Streamlit para look de app nativo */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* Tipografia maior e container centralizado */
html, body, [class*="css"]  { font-size: 17px; }
.block-container { padding-top: 0.6rem; padding-bottom: 4rem; max-width: 860px; }

/* Botões grandes e full-width para toque */
.stButton > button { width: 100%; padding: 0.9rem 1rem; font-weight: 700; border-radius: 12px; }

/* Inputs mais altos para toque confortável */
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] { min-height: 44px; }

/* Câmera e imagens responsivas */
[data-testid="stCameraInput"] video, [data-testid="stCameraInput"] img { width: 100% !important; height: auto !important; }

/* Tabelas com bordas suaves */
[data-testid="stDataFrame"] { border-radius: 12px; }

/* Rótulos com maior contraste */
label, .stCheckbox > label, .stRadio > label, .stSelectbox > label { font-weight: 600; }

/* Botão flutuante WhatsApp com área de toque maior */
a[href*="wa.me"] { font-size: 15px !important; padding: 12px 16px !important; }
</style>
""", unsafe_allow_html=True)

# ================================
# 2) SECRETS / VARIÁVEIS DE AMBIENTE / PASTAS
# ================================

# Senha admin (pode vir de st.secrets ou env); se não houver, usa default fraca (somente testes)
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", None)           # Tenta pegar do ambiente
try:
    if "ADMIN_PASSWORD" in st.secrets:                       # Se existir em st.secrets
        ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]        # Usa o secret
except Exception:
    pass                                                     # Ignora erro se secrets inexistente
if not ADMIN_PASSWORD:                                       # Se ainda não definido
    ADMIN_PASSWORD = "admin123"                              # Default (trocar em produção)

WHATSAPP_PHONE = os.getenv("WHATSAPP_ORG", "+9715xxxxxxxx")  # Telefone org (para botão flutuante)

DB_PATH = "registrations.db"                                 # Caminho do arquivo SQLite
UPLOAD_DIR = Path("uploads")                                 # Pasta para fotos
UPLOAD_DIR.mkdir(exist_ok=True)                              # Garante que a pasta exista

EVENT_DEFAULT_NAME = "ADSS Jiu-Jitsu Competition"            # Nome padrão do evento
EVENT_DEFAULT_REGION = "Al Dhafra – Abu Dhabi"               # Região padrão

# ================================
# 3) BANCO DE DADOS (SQLite via SQLAlchemy)
# ================================

engine = create_engine(f"sqlite:///{DB_PATH}", future=True)  # Cria/abre conexão com SQLite

def init_db():
    """Cria tabela principal e aplica migrações leves."""
    with engine.begin() as conn:                             # Abre transação
        # Criação da tabela se não existir
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
            approval_status TEXT,
            face_phash TEXT
        );
        """))
        # Migrações "idempotentes": tentamos adicionar colunas; se já existem, ignoramos
        for alter in [
            "ALTER TABLE registrations ADD COLUMN first_name TEXT",
            "ALTER TABLE registrations ADD COLUMN last_name TEXT",
            "ALTER TABLE registrations ADD COLUMN full_name_en TEXT",
            "ALTER TABLE registrations ADD COLUMN coach_phone TEXT",
            "ALTER TABLE registrations ADD COLUMN age_years INTEGER",
            "ALTER TABLE registrations ADD COLUMN approval_status TEXT",
            "ALTER TABLE registrations ADD COLUMN face_phash TEXT"
        ]:
            try:
                conn.execute(text(alter))                    # Tenta aplicar
            except Exception:
                pass                                         # Ignora erro (já aplicada)

init_db()                                                    # Executa inicialização ao carregar app

def insert_registration(row: dict):
    """Insere um registro (linha) no banco."""
    with engine.begin() as conn:                             # Abre transação
        cols = ",".join(row.keys())                          # Lista de colunas
        vals = ",".join([f":{k}" for k in row.keys()])       # Placeholders nomeados
        conn.execute(text(f"INSERT INTO registrations ({cols}) VALUES ({vals})"), row)  # Exec

def update_registration(reg_id: str, updates: dict):
    """Atualiza um registro pelo ID."""
    placeholders = ", ".join([f"{k} = :{k}" for k in updates.keys()])  # a= :a, b= :b...
    updates["id"] = reg_id                                             # Adiciona ID ao dict
    with engine.begin() as conn:                                       # Abre transação
        conn.execute(text(f"UPDATE registrations SET {placeholders} WHERE id = :id"), updates)

def fetch_all():
    """Retorna DataFrame com todas as inscrições, mais recentes primeiro."""
    with engine.begin() as conn:                          # Abre transação somente leitura
        df = pd.read_sql("SELECT * FROM registrations ORDER BY created_at DESC", conn)
    return df                                             # Retorna DF

def fetch_distinct_academies():
    """Lista academias já cadastradas (para dropdown)."""
    with engine.begin() as conn:                          # Abre transação leitura
        try:
            rows = conn.execute(text(
                "SELECT DISTINCT academy FROM registrations WHERE academy IS NOT NULL AND academy <> ''"
            )).fetchall()                                # Busca academias distintas
            return sorted({r[0] for r in rows})          # Conjunto -> lista ordenada
        except Exception:
            return []                                    # Em erro, retorna lista vazia

def count_by_category(category_value: str):
    """Conta quantos atletas existem em uma categoria específica."""
    with engine.begin() as conn:                          # Abre transação leitura
        try:
            row = conn.execute(text(
                "SELECT COUNT(*) FROM registrations WHERE category = :c"
            ), {"c": category_value}).fetchone()         # Conta linhas na categoria
            return int(row[0] if row else 0)             # Converte para int
        except Exception:
            return 0                                     # Em erro, assume 0

# ================================
# 4) UTIL – IMAGENS, FACE, PHASH, JPG DA INSCRIÇÃO
# ================================

def save_image(file, reg_id: str, suffix: str, max_size=(800, 800)) -> str:
    """Redimensiona e salva uma imagem enviada; retorna caminho salvo."""
    if file is None: return ""                            # Se não veio arquivo, retorna vazio
    image = Image.open(file)                              # Abre a imagem
    image.thumbnail(max_size)                             # Reduz tamanho para agilizar
    dest = UPLOAD_DIR / f"{reg_id}_{suffix}.jpg"         # Monta caminho destino
    image.save(dest, format="JPEG", quality=85)           # Salva em JPEG com qualidade 85
    return str(dest)                                      # Retorna caminho como string

def count_faces_in_image(file) -> int:
    """Detecta quantidade de rostos numa imagem; -1 se OpenCV indisponível."""
    if not FACE_DETECT_AVAILABLE: return -1               # Sem OpenCV: retorna -1 (tolerante)
    if file is None: return 0                             # Sem arquivo: 0 faces
    data = file.getvalue()                                # Pega bytes do arquivo
    arr = np.frombuffer(data, np.uint8)                   # Converte para array numpy
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)             # Decodifica para imagem OpenCV
    if img is None: return 0                              # Falha ao decodificar: 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)          # Converte para tons de cinza
    cascade = cv2.CascadeClassifier(                      # Carrega classificador Haar
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(80,80))  # Detecta faces
    return 0 if faces is None else len(faces)            # Retorna contagem

def crop_face_from_uploaded(file):
    """Recorta o rosto principal; se não detectar, retorna a imagem inteira (PIL RGB)."""
    if file is None: return None                          # Sem arquivo -> None
    data = file.getvalue()                                # Bytes da imagem
    img_pil = Image.open(BytesIO(data)).convert("RGB")    # Abre como PIL RGB
    if FACE_DETECT_AVAILABLE:                             # Se OpenCV disponível
        arr = np.frombuffer(data, np.uint8)               # Bytes -> array
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)       # Decodifica imagem
        if frame is not None:                             # Se deu certo
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)# Converte p/ cinza
            cascade = cv2.CascadeClassifier(              # Haar cascade
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(80,80))  # Detecta faces
            if faces is not None and len(faces)>0:        # Se encontrou
                x,y,w,h = max(faces, key=lambda r:r[2]*r[3]) # Maior face
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR->RGB
                img_pil = Image.fromarray(rgb).crop((x,y,x+w,y+h))  # Recorta rosto
    return img_pil                                        # Retorna PIL (rosto ou original)

def compute_phash(img_pil: Image.Image) -> str:
    """Computa perceptual hash (hex) de uma imagem PIL."""
    if img_pil is None: return ""                         # Sem imagem -> string vazia
    return str(imagehash.phash(img_pil))                  # Retorna pHash como texto

def phash_distance(a: str, b: str) -> int:
    """Distância de Hamming entre dois pHashes hex."""
    if not a or not b: return 1_000_000                   # Se faltou algum, retorna grande
    return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)  # Diferença (menor=mais parecido)

def generate_registration_jpg(row: dict, event_name: str, region: str, logo_file=None, width=1080, height=1350) -> bytes:
    """Gera um 'cartão' JPG da inscrição e retorna bytes para download."""
    from PIL import ImageDraw, ImageFont                   # Draw e fontes
    # Canvas branco
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)                             # Contexto de desenho

    # Tenta fontes "bonitas"; se não achar, usa default
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 60)
        font_sub   = ImageFont.truetype("DejaVuSans.ttf", 32)
        font_bold  = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
        font_text  = ImageFont.truetype("DejaVuSans.ttf", 34)
    except Exception:
        font_title = font_sub = font_bold = font_text = ImageFont.load_default()

    y = 40                                                # Cursor vertical inicial

    # Tenta colar logo (se enviado no sidebar e disponível)
    if logo_file is not None:
        try:
            lf = Image.open(logo_file).convert("RGBA")     # Abre logo com transparência
            maxw = int(width * 0.5)                        # Limita a 50% da largura
            ratio = maxw / lf.width                        # Fator de escala
            lf = lf.resize((maxw, int(lf.height * ratio))) # Redimensiona
            img.paste(lf, (int((width - lf.width) / 2), y), lf)  # Centraliza no topo
            y += lf.height + 20                            # Avança cursor vertical
        except Exception:
            pass                                           # Ignora falhas no logo

    # Título (nome do evento) centralizado
    title_text = event_name or "ADSS Jiu-Jitsu Competition"
    w_title, _ = draw.textsize(title_text, font=font_title)
    draw.text(((width - w_title) / 2, y), title_text, fill=(0, 0, 0), font=font_title)
    y += 70                                               # Avança cursor

    # Região (se houver)
    region_text = region or ""
    if region_text:
        w_reg, _ = draw.textsize(region_text, font=font_sub)
        draw.text(((width - w_reg) / 2, y), region_text, fill=(80, 80, 80), font=font_sub)
        y += 50

    # Linha divisória
    draw.line([(80, y), (width - 80, y)], fill=(230, 230, 230), width=3)
    y += 30

    # Foto de perfil (se salva)
    x_left = 80                                           # Margem esquerda
    photo_size = 320                                      # Tamanho quadrado
    profile_path = row.get("profile_photo_path", "")      # Caminho da foto de perfil
    if profile_path and os.path.exists(profile_path):     # Verifica existência
        try:
            p = Image.open(profile_path).convert("RGB").resize((photo_size, photo_size))  # Redimensiona
            img.paste(p, (x_left, y))                     # Cola na esquerda
        except Exception:
            pass

    # Função helper para linhas de texto
    x_text = x_left + photo_size + 40                     # Coluna de texto ao lado da foto
    def line(label, value):
        nonlocal y
        draw.text((x_text, y), f"{label}", fill=(0,0,0), font=font_bold); y += 40
        draw.text((x_text, y), f"{value}", fill=(40,40,40), font=font_text); y += 54

    # Dados principais
    name = f"{row.get('first_name','')} {row.get('last_name','')}".strip()
    line("Inscrição (ID):", row.get("id",""))
    line("Nome:", name or "-")
    line("Academia:", row.get("academy","-") or "-")
    line("Categoria:", row.get("category","-") or "-")
    line("Modalidade / Faixa:", f"{row.get('modality','-')} / {row.get('belt','-')}")
    line("Peso:", row.get("weight_class","-") or "-")
    line("Nacionalidade:", row.get("nationality","-") or "-")

    # Fecha bloco superior, garantindo altura mínima igual à foto
    y = max(y, 80 + photo_size + 30)
    draw.line([(80, y), (width - 80, y)], fill=(230, 230, 230), width=3)
    y += 30

    # Bloco inferior com contato/status
    col_x = 80                                            # Margem esquerda
    def kv(k, v):
        nonlocal y
        draw.text((col_x, y), k, fill=(0,0,0), font=font_bold); y += 40
        draw.text((col_x, y), v, fill=(40,40,40), font=font_text); y += 52

    kv("E-mail:", row.get("email","-") or "-")
    kv("Telefone:", row.get("phone","-") or "-")
    kv("Professor:", row.get("coach","-") or "-")
    kv("Status:", row.get("approval_status","Pending"))

    # QR code opcional (se lib 'qrcode' instalada)
    try:
        import qrcode                                  # Importa localmente (pode não existir)
        qr_data = f"Inscrição: {row.get('id','')}\nEvento: {event_name}"  # Conteúdo do QR
        qr_img = qrcode.make(qr_data).resize((240, 240))                   # Gera e redimensiona
        img.paste(qr_img, (width - 80 - 240, y - 280))                     # Cola no canto direito
    except Exception:
        pass

    # Rodapé com instrução
    foot = "Apresente este cartão na acreditação."
    w_foot, _ = draw.textsize(foot, font=font_sub)
    draw.text(((width - w_foot) / 2, height - 80), foot, fill=(120,120,120), font=font_sub)

    # Exporta para bytes (JPEG)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90, optimize=True)
    return buf.getvalue()                                 # Retorna bytes para st.download_button

# ================================
# 5) NORMALIZAÇÃO / OPÇÕES / CATEGORIAS
# ================================

def squeeze_spaces(s:str)->str:
    """Remove espaços duplicados e aparas."""
    return re.sub(r"\s+", " ", s or "").strip()

def title_capitalize(s:str)->str:
    """Primeira letra maiúscula em cada palavra (Title Case simples)."""
    return " ".join([w.capitalize() for w in squeeze_spaces(s).split(" ")])

def normalize_academy(s:str)->str:
    """Normaliza nome da academia (Title Case, sem espaços excedentes)."""
    return title_capitalize(s)

def format_phone_live(raw: str) -> str:
    """Mascara para 05_-___-____ sem callback (aplicada a cada render)."""
    raw = raw or ""
    digits = re.sub(r"\D", "", raw)
    if digits.startswith("9715"): digits = "05" + digits[-8:]  # Converte 9715xxxx -> 05xxxxxxxx
    if digits.startswith("5") and len(digits) <= 9: digits = "0" + digits  # Garante '05'
    digits = digits[:10]                                         # Limita a 10 dígitos
    if len(digits) <= 3: return digits
    if len(digits) <= 6: return f"{digits[:3]}-{digits[3:]}"
    return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"

def clean_and_format_phone_final(raw: str) -> str:
    """Valida e normaliza telefone ao submeter (mantém 052-123-4567)."""
    digits = re.sub(r"\D", "", raw or "")
    if digits.startswith("9715"): digits = "05" + digits[-8:]
    if digits.startswith("5") and len(digits) == 9: digits = "0" + digits
    digits = digits[:10]
    if len(digits) == 10 and digits.startswith("05"):
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    return raw

def translate_to_english(text_in: str) -> str:
    """Tradução automática para inglês (fallback para Title Case)."""
    try:
        from deep_translator import GoogleTranslator
        return title_capitalize(GoogleTranslator(source="auto", target="en").translate(text_in))
    except Exception:
        return title_capitalize(text_in)

# Opções fixas de competição
MODALIDADES = ["Gi", "No-Gi"]                             # Modalidades disponíveis
FAIXAS_GI = ["Branca","Cinza (Kids)","Amarela (Kids)","Laranja (Kids)","Verde (Kids)","Azul","Roxa","Marrom","Preta"]
FAIXAS_NOGI = ["Beginner","Intermediate","Advanced"]
PESOS_ADULT_MASC_GI = ["-56.5kg","-62kg","-69kg","-77kg","-85.5kg","-94kg","-102kg","+102kg"]
PESOS_ADULT_FEM_GI  = ["-48.5kg","-53.5kg","-58.5kg","-64kg","-69kg","-74kg","+74kg"]
PESOS_NOGI = ["-60kg","-66kg","-73kg","-80kg","-87kg","-94kg","-102kg","+102kg"]

def weight_options(mod, gen):
    """Retorna opções de peso conforme modalidade e gênero."""
    return PESOS_ADULT_MASC_GI if (mod=="Gi" and gen=="Masculino") else (PESOS_ADULT_FEM_GI if mod=="Gi" else PESOS_NOGI)

def belts_for(mod):
    """Retorna faixas conforme modalidade."""
    return FAIXAS_GI if mod=="Gi" else FAIXAS_NOGI

def get_country_list():
    """Lista de países (pycountry + extras; fallback enxuto)."""
    try:
        import pycountry
        countries = [c.name for c in pycountry.countries]
        extras = ["Hong Kong","Macau","Palestine","Kosovo"]
        return sorted(set(countries+extras))
    except Exception:
        return ["United Arab Emirates","Brazil","Portugal","United States","United Kingdom","Italy","Spain","France","Netherlands"]

def compute_age_year_based(year:int)->int:
    """Idade por ano-base (ano atual - ano de nascimento)."""
    if not year: return 0
    return max(0, dt.date.today().year - year)

def age_division_by_year(age:int)->str:
    """Divisão etária baseada na idade (Kids/Juvenile/Adult/Masters...)."""
    if age<=15: return "Kids"
    if 16<=age<=17: return "Juvenile"
    if 18<=age<=29: return "Adult"
    if 30<=age<=35: return "Master 1"
    if 36<=age<=40: return "Master 2"
    if 41<=age<=45: return "Master 3"
    return "Master 4+"

# ================================
# 6) ESTADO / NAVEGAÇÃO / RÓTULOS
# ================================

# Estados de navegação
if "screen" not in st.session_state: st.session_state["screen"]="welcome"            # Tela inicial
if "accepted_terms" not in st.session_state: st.session_state["accepted_terms"]=False# Aceite dos termos
if "event_name" not in st.session_state: st.session_state["event_name"]=EVENT_DEFAULT_NAME  # Nome do evento
if "region" not in st.session_state: st.session_state["region"]=EVENT_DEFAULT_REGION        # Região
if "errors" not in st.session_state: st.session_state["errors"]=set()                 # Campos com erro

def label(label_text: str, key: str, required: bool=False):
    """Rótulo com marcação vermelha se o campo tiver erro."""
    err = key in st.session_state["errors"]
    color = "#b91c1c" if err else "#111"
    star = " *" if required else ""
    st.markdown(f"<div style='font-weight:600;color:{color}'>{label_text}{star}</div>", unsafe_allow_html=True)

def whatsapp_button(phone: str):
    """Renderiza botão flutuante do WhatsApp (se número válido)."""
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
# 7) TELAS – FUNÇÕES DE VIEW
# ================================

def screen_welcome():
    """Tela 1 – Boas-vindas + guideline + aceite de termos."""
    st.markdown(f"<h1 style='text-align:center;margin-bottom:0'>{st.session_state['event_name']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;margin-top:4px;color:#777'>{st.session_state['region']}</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Bem-vindo(a)!")
    st.markdown("""
**Guideline da competição (resumo):**
- Chegar com antecedência mínima de 45 minutos.
- Documento de identificação oficial obrigatório.
- Uso de kimono/manguito conforme modalidade.
- Respeitar cronograma e chamadas de área.
- O organizador pode ajustar chaves/categorias em caso de WO.
- Ao se inscrever, você declara concordar com as regras do evento.
""")
    agree = st.checkbox("Li e **concordo** com os termos e condições", key="agree_terms")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state["event_name"] = st.text_input("Nome do evento", value=st.session_state["event_name"])
    with c2:
        st.session_state["region"] = st.text_input("Região / Cidade", value=st.session_state["region"])
    st.markdown("---")
    if st.button("Prosseguir"):                             # Botão para avançar
        if agree:                                           # Exige aceite
            st.session_state["accepted_terms"]=True         # Marca aceite
            st.session_state["screen"]="menu"               # Vai ao menu
        else:
            st.error("Você precisa concordar com os termos e condições para continuar.")  # Mensagem

def menu_button(label, key):
    """Botão do menu (com título acima)."""
    st.markdown(f"### {label}")
    return st.button("Abrir", key=key)

def screen_menu():
    """Tela 2 – Menu principal (novo, alterar, lista pública, admin)."""
    st.markdown(f"<h1 style='text-align:center;margin-bottom:0'>{st.session_state['event_name']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;margin-top:4px;color:#777'>{st.session_state['region']}</p>", unsafe_allow_html=True)
    st.markdown("---")
    if menu_button("Novo registro", "btn_new"): st.session_state["screen"]="new_registration"; return
    st.divider()
    if menu_button("Alterar registro", "btn_update"): st.session_state["screen"]="update_registration"; return
    st.caption("Validação por Face ID (selfie) ou número de inscrição.")
    st.divider()
    if menu_button("Lista de inscritos (pública)", "btn_public"): st.session_state["screen"]="public_list"; return
    st.divider()
    if menu_button("Gerenciamento da competição (admin)", "btn_admin"): st.session_state["screen"]="admin"; return

def screen_public_list():
    """Tela – Lista de inscritos (pública)."""
    st.title("Lista de Inscritos (pública)")
    df = fetch_all()
    if df.empty:
        st.info("Ainda não há inscrições.")
    else:
        # Seleciona/renomeia campos para exibição enxuta
        show = df[["id","first_name","last_name","academy","category","modality","belt","weight_class","approval_status"]].copy()
        show["Nome"] = (show["first_name"].fillna("") + " " + show["last_name"].fillna("")).str.strip()
        show = show.rename(columns={
            "id":"Inscrição","academy":"Academia","category":"Categoria",
            "modality":"Modalidade","belt":"Faixa","weight_class":"Peso","approval_status":"Status"
        })[["Inscrição","Nome","Academia","Categoria","Modalidade","Faixa","Peso","Status"]]
        st.dataframe(show, use_container_width=True, height=420)  # Altura fixa com scroll
    st.markdown("---")
    st.button("Voltar ao menu", on_click=lambda: st.session_state.update(screen="menu"))

def stats_view(embed: bool=False):
    """Bloco de estatísticas (pode ser usado no Admin)."""
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
    if {"category","modality"}.issubset(df.columns):
        st.subheader("Por modalidade")
        st.dataframe(df.pivot_table(index="category", columns="modality", values="id", aggfunc="count", fill_value=0), use_container_width=True)
    if {"category","belt"}.issubset(df.columns):
        st.subheader("Por faixa")
        st.dataframe(df.pivot_table(index="category", columns="belt", values="id", aggfunc="count", fill_value=0), use_container_width=True)

def admin_view():
    """Painel do organizador (lista + download JPG + estatísticas)."""
    st.subheader("Painel do Organizador")
    df = fetch_all()
    st.write(f"Total de inscrições: {len(df)}")
    if df.empty:
        st.info("Ainda não há inscrições."); return
    # Badge de "Pendente"
    st.markdown("""
        <style>.badge-pending{display:inline-block;padding:2px 8px;border-radius:999px;background:#F59E0B;
        color:#111;font-size:12px;font-weight:600;margin-left:8px;}</style>
    """, unsafe_allow_html=True)
    # Lista de inscritos como expansores
    for _, row in df.iterrows():
        name = f"{row.get('first_name','')} {row.get('last_name','')}".strip()
        status = (row.get("approval_status") or "Pending").strip()
        badge_html = "<span class='badge-pending'>Pendente de aprovação</span>" if status.lower()=="pending" else ""
        header_html = (name or row.get('id','(sem id)')) + " " + badge_html
        with st.expander(header_html, expanded=False):
            meta = {
                "ID": row.get("id",""),
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
            # Mostra as imagens se existirem
            if row.get("profile_photo_path") and Path(row["profile_photo_path"]).exists():
                st.image(row["profile_photo_path"], caption="Foto de Perfil", width=160)
            if row.get("id_doc_photo_path") and Path(row["id_doc_photo_path"]).exists():
                st.image(row["id_doc_photo_path"], caption="Documento", width=220)
            # Download do cartão em JPG
            jpg = generate_registration_jpg(row, st.session_state["event_name"], st.session_state["region"])
            st.download_button(
                "Baixar inscrição (JPG)",
                data=jpg,
                file_name=f"inscricao_{row['id']}.jpg",
                mime="image/jpeg",
                key=f"dl_admin_{row['id']}"
            )
    # Export CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Baixar CSV", data=csv, file_name="inscricoes.csv", mime="text/csv")
    st.markdown("---"); stats_view(embed=True)  # Estatísticas embutidas

def screen_admin():
    """Tela – Login admin + painel."""
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
    st.button("Voltar ao menu", on_click=lambda: st.session_state.update(screen="menu"))

def render_edit_form(best_row):
    """Formulário para editar inscrição (reaproveitado em atualização por selfie/ID)."""
    st.markdown("---"); st.subheader("Editar dados")
    with st.form(f"edit_{best_row['id']}", clear_on_submit=False):
        ef_first_name = st.text_input("Nome", value=best_row.get("first_name",""))
        ef_last_name  = st.text_input("Sobrenome", value=best_row.get("last_name",""))
        ef_email      = st.text_input("E-mail", value=best_row.get("email",""))
        ef_phone      = st.text_input("Telefone (05_-___-____)", value=best_row.get("phone",""))
        ef_nationality= st.text_input("Nacionalidade", value=best_row.get("nationality",""))
        ef_gender     = st.selectbox("Gênero", ["Masculino","Feminino"], index=0 if best_row.get("gender","Masculino")=="Masculino" else 1)
        ef_modality   = st.selectbox("Modalidade", MODALIDADES, index=0 if best_row.get("modality","Gi")=="Gi" else 1)
        belt_opts     = belts_for(ef_modality)
        ef_belt       = st.selectbox("Faixa", belt_opts, index=belt_opts.index(best_row.get("belt", belt_opts[0])) if best_row.get("belt","") in belt_opts else 0)
        weight_opts   = weight_options(ef_modality, ef_gender)
        ef_weight     = st.selectbox("Peso", weight_opts, index=weight_opts.index(best_row.get("weight_class", weight_opts[0])) if best_row.get("weight_class","") in weight_opts else 0)
        ef_academy    = st.text_input("Academia", value=best_row.get("academy",""))
        ef_coach      = st.text_input("Professor/Coach", value=best_row.get("coach",""))
        ef_coach_phone= st.text_input("Telefone do Professor", value=best_row.get("coach_phone",""))
        st.markdown("Opcional: nova foto de perfil")
        new_profile = st.camera_input("Nova foto de perfil") or st.file_uploader("Ou enviar arquivo", type=["jpg","jpeg","png"], key=f"file_{best_row['id']}")
        submitted_update = st.form_submit_button("Salvar alterações")
    # Download do cartão (estado atual)
    jpg_now = generate_registration_jpg(best_row, st.session_state["event_name"], st.session_state["region"])
    st.download_button(
        "Baixar inscrição (JPG)",
        data=jpg_now,
        file_name=f"inscricao_{best_row['id']}.jpg",
        mime="image/jpeg",
        key=f"dl_edit_{best_row['id']}"
    )
    if submitted_update:
        # Normalizações finais
        ef_first_name = title_capitalize(ef_first_name); ef_last_name = title_capitalize(ef_last_name)
        ef_academy = normalize_academy(ef_academy)
        ef_phone = clean_and_format_phone_final(ef_phone)
        ef_coach_phone = clean_and_format_phone_final(ef_coach_phone)
        updates = {
            "first_name": ef_first_name, "last_name": ef_last_name,
            "full_name_en": translate_to_english(f"{ef_first_name} {ef_last_name}"),
            "email": (ef_email or "").strip(), "phone": ef_phone,
            "nationality": ef_nationality, "gender": ef_gender,
            "academy": ef_academy, "coach": title_capitalize(ef_coach),
            "coach_phone": ef_coach_phone, "modality": ef_modality,
            "belt": ef_belt, "weight_class": ef_weight,
        }
        # Se trocar a foto, atualiza arquivo e pHash
        if new_profile is not None:
            reg_id = best_row["id"]
            new_path = save_image(new_profile, reg_id, "profile")
            updates["profile_photo_path"] = new_path
            crop = crop_face_from_uploaded(new_profile)
            updates["face_phash"] = compute_phash(crop)
        try:
            update_registration(best_row["id"], updates)   # Salva no banco
            st.success("Registro atualizado com sucesso.")
        except Exception as e:
            st.error(f"Erro ao atualizar: {e}")

def screen_update_registration():
    """Tela – Alterar registro (por Face ID ou por ID)."""
    st.title("Alterar Inscrição – Identificação")
    df = fetch_all()
    if df.empty:
        st.info("Não há inscrições para alterar."); 
        st.button("Voltar", on_click=lambda: st.session_state.update(screen="menu"))
        return
    tab1, tab2 = st.tabs(["Face ID (selfie)", "ID da inscrição"])
    # --- Aba 1: Face ID ---
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
    # --- Aba 2: por ID ---
    with tab2:
        regid = st.text_input("Digite o número da inscrição (ID)")
        if st.button("Localizar"):
            row = df[df["id"]==regid]
            if row.empty:
                st.error("ID não encontrado.")
            else:
                render_edit_form(row.iloc[0])
    st.markdown("---")
    st.button("Voltar ao menu", on_click=lambda: st.session_state.update(screen="menu"))

def screen_new_registration():
    """Tela – Formulário de novo registro (com validações, máscara, fotos e JPG)."""
    # Pré-máscara (aplica formato a cada render, sem callbacks)
    if "phone" in st.session_state:
        st.session_state["phone"] = format_phone_live(st.session_state["phone"])
    if "coach_phone" in st.session_state:
        st.session_state["coach_phone"] = format_phone_live(st.session_state["coach_phone"])

    st.title("Formulário de Inscrição")
    st.caption("Preencha seus dados. Campos com * são obrigatórios.")

    with st.form("registration_form", clear_on_submit=False):
        # Duas colunas (empilham no mobile)
        colA, colB = st.columns(2)

        with colA:
            label("Nome", "first_name", required=True)
            first_name_raw = st.text_input("", key="first_name", label_visibility="collapsed", max_chars=60, placeholder="Ex.: João")

            label("Sobrenome", "last_name", required=True)
            last_name_raw  = st.text_input("", key="last_name", label_visibility="collapsed", max_chars=60, placeholder="Ex.: Silva")

            label("E-mail", "email", required=True)
            email = st.text_input("", key="email", label_visibility="collapsed", placeholder="email@exemplo.com")

            label("Telefone/WhatsApp (formato 05_-___-____)", "phone", required=True)
            phone_input = st.text_input("", key="phone", label_visibility="collapsed", placeholder="052-123-4567")

            label("Nacionalidade", "nationality", required=True)
            nationality = st.selectbox("", get_country_list(), key="nationality", label_visibility="collapsed")

            label("Gênero", "gender", required=True)
            gender = st.selectbox("", ["Masculino","Feminino"], key="gender", label_visibility="collapsed")

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

            # Academias: lista existente + opção "não está na lista"
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
            coach_phone_input = st.text_input("", key="coach_phone", label_visibility="collapsed", placeholder="052-123-4567")

        # Fotos obrigatórias
        st.subheader("Fotos obrigatórias")
        label("Foto de Perfil (rosto visível)", "profile_img", required=True)
        profile_img = st.camera_input("", key="profile_img", label_visibility="collapsed")

        label("Documento de Identificação (frente)", "id_doc_img", required=True)
        id_doc_img = st.camera_input("", key="id_doc_img", label_visibility="collapsed")

        # Competição
        st.subheader("Informações de Competição")
        label("Modalidade", "modality", required=True)
        modality = st.selectbox("", MODALIDADES, key="modality", label_visibility="collapsed")

        label("Faixa", "belt", required=True)
        belt = st.selectbox("", belts_for(modality), key="belt", label_visibility="collapsed")

        label("Peso", "weight_class", required=True)
        weight_class = st.selectbox("", weight_options(modality, gender), key="weight_class", label_visibility="collapsed")

        # Consentimento
        st.subheader("Termo de Consentimento")
        st.write("Declaro estar apto(a) a participar e concordo com as regras do evento.")
        consent = st.checkbox("Eu li e concordo com o termo de consentimento*", key="consent")

        # Observações
        notes = st.text_area("Observações (opcional)", height=80, key="notes")

        # Botão de envio do form
        submitted = st.form_submit_button("Enviar inscrição")

    # Processamento do submit
    if submitted:
        errors = set()                                     # Conjunto para chaves com erro
        # Checa obrigatórios
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

        # Checagem de rosto (tolerante)
        if "profile_img" not in errors and st.session_state.get("profile_img") is not None:
            faces_profile = count_faces_in_image(st.session_state.get("profile_img"))
            if faces_profile == -1:
                st.warning("Validação automática de rosto indisponível no ambiente atual. Prosseguindo sem essa checagem.")
            else:
                if faces_profile == 0:
                    errors.add("profile_img"); st.error("A foto de perfil não parece conter um rosto. Envie uma foto clara do rosto.")
                elif faces_profile > 1:
                    errors.add("profile_img"); st.error("Detectamos mais de uma pessoa na foto de perfil. Envie uma foto só do atleta.")

        # Atualiza estado de erros para destacar rótulos em vermelho
        st.session_state["errors"] = errors

        # Se houver erros, interrompe
        if errors:
            st.error("Há campos obrigatórios faltando. Eles estão destacados em vermelho acima.")
            return

        # Normalizações finais
        first_name = title_capitalize(st.session_state["first_name"])
        last_name  = title_capitalize(st.session_state["last_name"])
        full_name_en = translate_to_english(f"{first_name} {last_name}")
        phone = clean_and_format_phone_final(st.session_state["phone"])
        coach_phone = clean_and_format_phone_final(st.session_state.get("coach_phone","")) if st.session_state.get("coach_phone") else ""
        if st.session_state["academy_choice"] == "Minha academia não está na lista":
            academy = normalize_academy(st.session_state["academy_other"])
        else:
            academy = normalize_academy(st.session_state["academy_choice"])
        # Data de nascimento
        try:
            dob_date = dt.date(int(st.session_state["dob_year"]), int(st.session_state["dob_month"]), int(st.session_state["dob_day"]))
            dob_iso = dob_date.isoformat()
        except Exception:
            st.error("Data de nascimento inválida. Verifique dia, mês e ano.")
            return

        # Cálculo de idade (ano-base) e categoria
        age_years = compute_age_year_based(int(st.session_state["dob_year"]))
        age_div = age_division_by_year(age_years)
        category = age_div

        # Salva imagens
        reg_id = str(uuid.uuid4())[:8].upper()
        profile_photo_path = save_image(st.session_state["profile_img"], reg_id, "profile")
        id_doc_photo_path = save_image(st.session_state["id_doc_img"], reg_id, "id_doc")

        # pHash do rosto (ou imagem inteira) para futura identificação por selfie
        face_crop_img = crop_face_from_uploaded(st.session_state["profile_img"])
        face_phash = compute_phash(face_crop_img)

        # Monta linha a inserir
        row = {
            "id": reg_id,
            "created_at": dt.datetime.utcnow().isoformat(),
            "event_name": st.session_state["event_name"],
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
            "region": st.session_state["region"],
            "modality": st.session_state["modality"],
            "belt": st.session_state["belt"],
            "weight_class": st.session_state["weight_class"],
            "category": category,
            "consent": 1,
            "notes": (st.session_state.get("notes") or "").strip(),
            "profile_photo_path": profile_photo_path,
            "id_doc_photo_path": id_doc_photo_path,
            "approval_status": "Pending",
            "face_phash": face_phash
        }

        # Salva no banco + feedback + contagem de categoria + download JPG
        try:
            insert_registration(row)                         # Insere no DB
            st.success(f"Inscrição enviada com sucesso. ID: {reg_id}")  # Mensagem OK
            st.info("Status da sua inscrição: Pendente de aprovação.")  # Status
            total_cat = count_by_category(category)          # Conta categoria
            st.info(f"Atletas já inscritos na categoria '{category}': {total_cat}")  # Mostra contagem

            # Download do cartão em JPG
            jpg_bytes = generate_registration_jpg(row, st.session_state["event_name"], st.session_state["region"])
            st.download_button(
                "Baixar inscrição (JPG)", data=jpg_bytes,
                file_name=f"inscricao_{row['id']}.jpg", mime="image/jpeg"
            )

            # Link rápido para WhatsApp
            if WHATSAPP_PHONE.strip():
                msg = f"Olá! Minha inscrição do evento {st.session_state['event_name']} foi enviada. Meu ID é {reg_id}."
                phone_clean = WHATSAPP_PHONE.replace("+", "").replace(" ", "").replace("-", "")
                wa_url = f"https://wa.me/{phone_clean}?text=" + urllib.parse.quote(msg)
                st.markdown(f"[Falar com a organização no WhatsApp]({wa_url})")

        except Exception as e:
            st.error(f"Erro ao salvar inscrição: {e}")

def screen_admin_gate():
    """Porta de entrada do Admin (senha + navegação de volta)."""
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
    st.button("Voltar ao menu", on_click=lambda: st.session_state.update(screen="menu"))

# ================================
# 8) ROTEAMENTO DE TELAS
# ================================

# Exibe botão de WhatsApp (flutuante) em todas as telas
whatsapp_button(WHATSAPP_PHONE)

# Navega conforme a tela atual e aceite de termos
if st.session_state["screen"] == "welcome":
    screen_welcome()
elif not st.session_state.get("accepted_terms", False):
    # Se não aceitou termos, força voltar à tela de boas-vindas
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
    # Fallback: volta ao menu se o estado for desconhecido
    st.session_state["screen"] = "menu"
    screen_menu()
