import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time

# Try to import xgboost, handle if missing
try:
    import xgboost
except ImportError:
    xgboost = None

# Page Config
st.set_page_config(
    page_title="🥛 MilkGuard Pro | AI Quality Analyzer",
    page_icon="🥛",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================================
# STUNNING INDUSTRY-LEVEL CSS
# ================================
st.markdown("""
<style>
    /* ===== GOOGLE FONTS ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&display=swap');
    
    /* ===== GLOBAL RESET & BASE ===== */
    * {
        font-family: 'Inter', sans-serif !important;
        box-sizing: border-box;
    }
    
    .stApp {
        background: linear-gradient(160deg, #0D1B2A 0%, #1B263B 30%, #415A77 60%, #778DA9 100%);
        min-height: 100vh;
    }
    
    /* ===== HIDE STREAMLIT DEFAULTS ===== */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    div[data-testid="stToolbar"] {visibility: hidden;}
    
    /* ===== GLASSMORPHISM CARDS ===== */
    .glass-ultra {
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.1) 0%, 
            rgba(255, 255, 255, 0.05) 100%);
        backdrop-filter: blur(40px) saturate(180%);
        -webkit-backdrop-filter: blur(40px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 32px;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            inset 0 -1px 0 rgba(0, 0, 0, 0.1);
        padding: 2.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .glass-ultra::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, 
            transparent 0%, 
            rgba(255, 255, 255, 0.4) 50%, 
            transparent 100%);
    }
    
    .glass-input {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 2rem;
        transition: all 0.4s ease;
    }
    
    .glass-input:hover {
        background: rgba(255, 255, 255, 0.12);
        border-color: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
    }
    
    /* ===== HERO HEADER ===== */
    .hero-section {
        text-align: center;
        padding: 4rem 2rem;
        position: relative;
    }
    
    .hero-icon {
        font-size: 5rem;
        display: inline-block;
        animation: float 4s ease-in-out infinite;
        filter: drop-shadow(0 10px 30px rgba(255, 255, 255, 0.2));
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0) rotate(-5deg); }
        50% { transform: translateY(-20px) rotate(5deg); }
    }
    
    .hero-title {
        font-family: 'Playfair Display', serif !important;
        font-size: 4.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 0%, #E0E1DD 50%, #94D2BD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 1rem 0 0.5rem 0;
        letter-spacing: -2px;
        line-height: 1.1;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 400;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    
    .hero-tagline {
        font-size: 1rem;
        color: #94D2BD;
        font-weight: 500;
        margin-top: 1rem;
    }
    
    /* ===== SECTION TITLES ===== */
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .section-title::before {
        content: '';
        width: 6px;
        height: 36px;
        background: linear-gradient(180deg, #94D2BD 0%, #52B788 100%);
        border-radius: 3px;
    }
    
    .section-subtitle {
        font-size: 1.1rem;
        font-weight: 600;
        color: #E0E1DD;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(148, 210, 189, 0.3);
    }
    
    /* ===== INPUT STYLING ===== */
    .stSlider label, .stSlider p {
        color: #E0E1DD !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #52B788 0%, #94D2BD 100%) !important;
    }
    
    .stRadio label, .stRadio p {
        color: #E0E1DD !important;
        font-weight: 600 !important;
    }
    
    .stRadio > div > div > label > div {
        color: #FFFFFF !important;
    }
    
    /* ===== PREMIUM BUTTON ===== */
    .stButton > button {
        background: linear-gradient(135deg, #52B788 0%, #2D6A4F 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 1.2rem 3rem !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        box-shadow: 
            0 10px 40px rgba(82, 183, 136, 0.4),
            0 4px 12px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) scale(1.02) !important;
        box-shadow: 
            0 20px 60px rgba(82, 183, 136, 0.5),
            0 8px 20px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
    }
    
    /* ===== RESULT CARDS ===== */
    .result-card {
        text-align: center;
        padding: 3rem;
        border-radius: 32px;
        animation: resultPop 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    @keyframes resultPop {
        0% { opacity: 0; transform: scale(0.8) translateY(20px); }
        100% { opacity: 1; transform: scale(1) translateY(0); }
    }
    
    .result-high {
        background: linear-gradient(135deg, rgba(82, 183, 136, 0.3) 0%, rgba(45, 106, 79, 0.2) 100%);
        border: 2px solid rgba(82, 183, 136, 0.5);
    }
    
    .result-medium {
        background: linear-gradient(135deg, rgba(255, 183, 77, 0.3) 0%, rgba(255, 145, 0, 0.2) 100%);
        border: 2px solid rgba(255, 183, 77, 0.5);
    }
    
    .result-low {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.3) 0%, rgba(185, 28, 28, 0.2) 100%);
        border: 2px solid rgba(239, 68, 68, 0.5);
    }
    
    .result-emoji {
        font-size: 5rem;
        display: block;
        margin-bottom: 1rem;
    }
    
    .result-text {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    
    .result-desc {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.8);
        margin-top: 1rem;
    }
    
    /* ===== METRIC CARDS ===== */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.5rem;
        margin-top: 2rem;
    }
    
    .metric-item {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-item:hover {
        background: rgba(255, 255, 255, 0.15);
        transform: translateY(-4px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #94D2BD;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.6);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }
    
    /* ===== DEVELOPER INFO CARD ===== */
    .dev-card {
        background: linear-gradient(135deg, 
            rgba(148, 210, 189, 0.1) 0%, 
            rgba(82, 183, 136, 0.05) 100%);
        backdrop-filter: blur(30px);
        border: 1px solid rgba(148, 210, 189, 0.2);
        border-radius: 24px;
        padding: 2rem;
        text-align: center;
        margin-top: 3rem;
    }
    
    .dev-avatar {
        width: 80px;
        height: 80px;
        background: linear-gradient(135deg, #52B788 0%, #2D6A4F 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1rem auto;
        font-size: 2rem;
        color: white;
        box-shadow: 0 10px 30px rgba(82, 183, 136, 0.3);
    }
    
    .dev-name {
        font-size: 1.5rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.5rem;
    }
    
    .dev-role {
        font-size: 1rem;
        color: #94D2BD;
        margin-bottom: 1rem;
    }
    
    .dev-info {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.7);
        line-height: 1.6;
    }
    
    /* ===== TECH STACK BADGES ===== */
    .tech-stack {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 1.5rem;
        flex-wrap: wrap;
    }
    
    .tech-badge {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 50px;
        padding: 0.5rem 1.2rem;
        font-size: 0.85rem;
        color: #E0E1DD;
        font-weight: 500;
    }
    
    /* ===== FOOTER ===== */
    .custom-footer {
        text-align: center;
        padding: 3rem 2rem;
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.9rem;
        margin-top: 4rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ================================
# LOAD MODEL
# ================================
@st.cache_resource
def load_model():
    try:
        with open('milk_quality_xgb_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        return None

model = load_model()

# ================================
# HERO HEADER
# ================================
st.markdown("""
<div class="glass-ultra hero-section">
    <span class="hero-icon">🥛</span>
    <h1 class="hero-title">MilkGuard Pro</h1>
    <p class="hero-subtitle">AI-Powered Quality Grading System</p>
    <p class="hero-tagline">Advanced Machine Learning • Real-Time Analysis • Industry Grade</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ================================
# MAIN CONTENT
# ================================
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("""
    <div class="glass-input">
        <div class="section-title">Sample Parameters</div>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="section-subtitle">🧪 Chemical Properties</p>', unsafe_allow_html=True)
    ph = st.slider("pH Level", min_value=3.0, max_value=9.5, value=6.6, step=0.1,
                   help="Normal milk pH: 6.5 - 6.7")
    
    st.markdown('<p class="section-subtitle">🌡️ Physical Properties</p>', unsafe_allow_html=True)
    temperature = st.slider("Temperature (°C)", min_value=34, max_value=90, value=35,
                            help="Sample temperature")
    colour = st.slider("Colour Index", min_value=240, max_value=255, value=250,
                       help="Visual measurement")
    
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div class="glass-input">
        <div class="section-title">Quality Indicators</div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown('<p class="section-subtitle">👅 Taste</p>', unsafe_allow_html=True)
        taste = st.radio("taste_label", [1, 0], 
                        format_func=lambda x: "✅ Good" if x==1 else "❌ Bad", 
                        key="taste", horizontal=True, label_visibility="collapsed")
        
        st.markdown('<p class="section-subtitle">💨 Odor</p>', unsafe_allow_html=True)
        odor = st.radio("odor_label", [1, 0], 
                       format_func=lambda x: "✅ Good" if x==1 else "❌ Bad", 
                       key="odor", horizontal=True, label_visibility="collapsed")
    
    with c2:
        st.markdown('<p class="section-subtitle">🧈 Fat Content</p>', unsafe_allow_html=True)
        fat = st.radio("fat_label", [1, 0], 
                      format_func=lambda x: "✅ Optimal" if x==1 else "❌ Not Optimal", 
                      key="fat", horizontal=True, label_visibility="collapsed")
        
        st.markdown('<p class="section-subtitle">💧 Turbidity</p>', unsafe_allow_html=True)
        turbidity = st.radio("turbidity_label", [1, 0], 
                            format_func=lambda x: "✅ Low" if x==1 else "❌ High", 
                            key="turbidity", horizontal=True, label_visibility="collapsed")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ================================
# ANALYZE BUTTON
# ================================
st.markdown("<br>", unsafe_allow_html=True)
col_btn = st.columns([1, 2, 1])
with col_btn[1]:
    analyze_clicked = st.button("🔬 ANALYZE SAMPLE QUALITY", use_container_width=True)

# ================================
# RESULTS
# ================================
if analyze_clicked:
    if model is None:
        st.error("⚠️ Model not found! Please run: `./venv/bin/python generate_model.py`")
    else:
        with st.spinner("🔄 AI analyzing milk composition..."):
            time.sleep(1.5)
        
        input_data = np.array([[ph, temperature, taste, odor, fat, turbidity, colour]])
        prediction = model.predict(input_data)[0]
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if prediction == 2:
            result_class = "result-high"
            result_emoji = "🏆"
            result_text = "PREMIUM QUALITY"
            result_color = "#52B788"
            result_desc = "Exceptional! This milk sample exceeds all quality standards."
            st.balloons()
        elif prediction == 1:
            result_class = "result-medium"
            result_emoji = "✓"
            result_text = "ACCEPTABLE QUALITY"
            result_color = "#FFB74D"
            result_desc = "This sample meets standard requirements."
        else:
            result_class = "result-low"
            result_emoji = "⚠️"
            result_text = "BELOW STANDARD"
            result_color = "#EF4444"
            result_desc = "This sample requires further testing."
        
        st.markdown(f"""
        <div class="glass-ultra result-card {result_class}">
            <span class="result-emoji">{result_emoji}</span>
            <h2 class="result-text" style="color: {result_color}">{result_text}</h2>
            <p class="result-desc">{result_desc}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        st.markdown("""
        <div class="glass-input" style="margin-top: 2rem;">
            <div class="section-title">📊 Analysis Summary</div>
            <div class="metric-grid">
        """, unsafe_allow_html=True)
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-item"><div class="metric-value">{ph}</div><div class="metric-label">pH Level</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-item"><div class="metric-value">{temperature}°C</div><div class="metric-label">Temperature</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-item"><div class="metric-value">{colour}</div><div class="metric-label">Colour</div></div>', unsafe_allow_html=True)
        with m4:
            score = (taste + odor + fat + (1-turbidity)) / 4 * 100
            st.markdown(f'<div class="metric-item"><div class="metric-value">{score:.0f}%</div><div class="metric-label">Quality Score</div></div>', unsafe_allow_html=True)

# ================================
# DEVELOPER INFO
# ================================
st.markdown("""
<div class="dev-card">
    <div class="dev-avatar">🧑‍💻</div>
    <h3 class="dev-name">Partha Sarathi R</h3>
    <p class="dev-role">Machine Learning Engineer & Developer</p>
    <p class="dev-info">
        Computer Science Student | AI/ML Enthusiast<br>
        Building intelligent solutions for real-world problems
    </p>
    <div class="tech-stack">
        <span class="tech-badge">🐍 Python</span>
        <span class="tech-badge">🤖 XGBoost</span>
        <span class="tech-badge">📊 Scikit-Learn</span>
        <span class="tech-badge">🎨 Streamlit</span>
        <span class="tech-badge">📈 Pandas</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ================================
# FOOTER
# ================================
st.markdown("""
<div class="custom-footer">
    <p><strong>MilkGuard Pro</strong> • Powered by XGBoost Machine Learning</p>
    <p>© 2026 Partha Sarathi R. All rights reserved.</p>
    <p style="margin-top: 1rem; font-size: 0.8rem;">
        🎓 College Project • Machine Learning Classification • Quality Assurance System
    </p>
</div>
""", unsafe_allow_html=True)
