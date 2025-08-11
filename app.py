# ================================
# app.py – ADSS Jiu-Jitsu Competition (Mobile-Optimized)
# ================================

# [Previous imports remain the same...]

# ================================
# 1) CONFIG GLOBAL
# ================================
st.set_page_config(page_title="ADSS Jiu-Jitsu Competition", layout="centered")  # Changed from wide to centered
st.markdown("""
<style>
#MainMenu, header, footer { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 100%; }
@media (min-width: 768px) {
    .block-container { max-width: 860px; }
}
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

/* Mobile-specific styles */
@media (max-width: 768px) {
    .stButton > button { padding: 0.9rem 1rem; font-size: 0.95rem; }
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] { min-height: 50px; font-size: 16px; }
    .card { flex-direction: column; }
    .card img { width: 100%; max-height: 200px; object-fit: cover; }
    .element-container { margin-bottom: 1rem; }
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column"] > div > div > div { margin-bottom: 1rem; }
}

/* Better camera capture button */
.camera-button {
    position: relative;
    display: flex;
    justify-content: center;
    margin: 1rem 0;
}
.camera-button button {
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 50%;
    width: 70px;
    height: 70px;
    font-size: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    cursor: pointer;
}
.camera-button button:hover {
    background-color: #45a049;
}

/* Mobile form layout */
@media (max-width: 768px) {
    div[data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 0;
    }
    div[data-testid="stHorizontalBlock"] > div {
        width: 100% !important;
        margin-bottom: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)

# [Rest of the code remains the same until the webrtc_capture_block function...]

# ================================
# 8) WEBCAM (streamlit-webrtc) – foco contínuo + captura
# ================================
RTC_CFG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class SnapshotProcessor(VideoProcessorBase):
    def __init__(self):
        self.latest_frame = None
    def recv(self, frame):
        self.latest_frame = frame.to_ndarray(format="bgr24")
        return frame

def webrtc_capture_block(title: str, key_prefix: str, facing="user"):
    """
    Renderiza um bloco de câmera via streamlit-webrtc com foco contínuo (quando suportado).
    Retorna uma PIL Image quando o usuário clica "Capturar foto".
    """
    st.caption(title)
    if not WEBRTC_AVAILABLE:
        return None, st.camera_input(title + " (fallback)")
    
    # Mobile-optimized constraints
    constraints = {
        "video": {
            "facingMode": facing,          # frontal (selfie) ou "environment"
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
            "frameRate": {"ideal": 30},
            # Dica para autofocus (browsers que suportam):
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
    if ctx and ctx.video_processor:
        # Mobile-optimized capture button
        st.markdown("""
        <div class="camera-button">
            <button>📷</button>
        </div>
        """, unsafe_allow_html=True)
        
        # Using a hidden button to trigger capture
        capture_button = st.empty()
        if capture_button.button(f"Capturar foto – {title}", key=f"btn_{key_prefix}"):
            frame = ctx.video_processor.latest_frame
            if frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if FACE_DETECT_AVAILABLE else frame[..., ::-1]
                snap = Image.fromarray(rgb)
                # Remove the capture button after successful capture
                capture_button.empty()
    
    return snap, None  # (PIL, fallback_uploaded)

# [Rest of the code remains the same...]
