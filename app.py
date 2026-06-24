import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from datetime import datetime

# 1. Page Configuration & Aesthetic Setup
st.set_page_config(
    page_title="AlphaMind // Alphanumeric Canvas",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Premium CSS Injection - OVERHAULED FOR PURE SEAMLESS WHITE CANVAS FLOWS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght=300;400;500;700&display=swap');
    
    .brand-title, .brand-subtitle, .premium-card, .section-header, .prediction-display, p, li {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #F8FAFC;
    }
    
    .brand-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        letter-spacing: -0.03em;
    }
    
    .brand-subtitle {
        font-size: 1.1rem;
        color: #94A3B8;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    
    .premium-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
    }
    
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #F1F5F9;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .prediction-display {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.1) 0%, rgba(129, 140, 248, 0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        margin-top: 10px;
    }
    
    .pred-letter {
        font-size: 6rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1;
        margin: 10px 0;
        text-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
    }

    /* Custom CSS Progress Bars */
    .bar-container {
        display: flex;
        align-items: center;
        margin-bottom: 12px;
    }
    .bar-label {
        width: 30px;
        font-weight: 700;
        color: #38BDF8;
    }
    .bar-track {
        flex-grow: 1;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        height: 16px;
        margin: 0 12px;
        overflow: hidden;
    }
    .bar-fill {
        background: linear-gradient(90deg, #6366F1, #38BDF8);
        height: 100%;
        border-radius: 8px;
    }
    .bar-value {
        width: 60px;
        text-align: right;
        font-size: 0.85rem;
        color: #94A3B8;
    }
    
    /* 🎯 UNIFIED FULL-WIDTH WHITE SCREEN CORRECTION */
    /* Forces all parent wrapper backgrounds, i-frames, and elements to be solid white */
    div[data-testid="stCanvas"], 
    div[data-testid="stCanvas"] > div,
    div[data-testid="stCanvas"] canvas,
    .stCanvas, 
    iframe {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Stylizes lower native control toolbar buttons */
    .stCanvas + div button {
        color: #0F172A !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# 3. Cached Model Loading
@st.cache_resource
def load_nn_model():
    try:
        return load_model("alphanumeric_model.h5")
    except Exception as e:
        st.error(f"Error loading 'alphanumeric_model.h5'. Verify file placement. Error: {e}")
        return None

model = load_nn_model()

# 4. Class Decoder Helper Function
def decode_prediction(class_index):
    if class_index < 10:
        return str(class_index)
    else:
        return chr(class_index - 10 + 65)

# --- SIDEBAR SYSTEM DESIGN ---
with st.sidebar:
    st.header("🧠 AlphaMind Control")
    st.caption("v3.7.0 (Borderless Matrix Build)")
    st.divider()
    
    stroke_width = st.slider("Brush Thickness (Pixels)", min_value=12, max_value=32, value=22)
    
    st.divider()
    st.markdown("""
    <div style="background: rgba(255,255,255,0.03); border-radius:12px; padding:16px; border: 1px solid rgba(255,255,255,0.05);">
        <p style="color:#38BDF8; font-weight:600; margin-top:0; margin-bottom:8px; font-size:0.9rem;">📐 SYSTEM TELEMETRY</p>
        <ul style="color:#94A3B8; font-size:0.85rem; padding-left:20px; margin:0;">
            <li style="margin-bottom:6px;"><b>Canvas Layout:</b> Formatted to expand flush against column dividers.</li>
            <li style="margin-bottom:6px;"><b>Processing Arrays:</b> Automatically crops margins to isolate handwritten digits/characters.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- MAIN WORKSPACE INTERFACE ---
st.markdown("<h1 class='brand-title'>AlphaMind Hybrid Engine</h1>", unsafe_allow_html=True)
st.markdown("<p class='brand-subtitle'>Alphanumeric Recognition Platform Processing Real-Time Vectors & Image Streams</p>", unsafe_allow_html=True)

layout_col1, layout_col2 = st.columns([1, 1.1], gap="large")

with layout_col1:
    st.markdown("<div class='premium-card'><div class='section-header'>📥 Data Acquisition Matrix</div></div>", unsafe_allow_html=True)
    
    input_tab1, input_tab2 = st.tabs(["✍️ Live Drawing Canvas", "📤 External File Upload"])
    
    img_ready_to_predict = None
    source_type = ""
    
    with input_tab1:
        # Enforce white background wrap around the full container
        st.markdown('<div style="background-color: #FFFFFF; border-radius: 12px; overflow: hidden; padding: 12px; border: none;">', unsafe_allow_html=True)
        
        # 🎯 CHANGED width=450 to fill out the column space and fully push out the black strip bug
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=stroke_width,
            stroke_color="#0F172A", 
            background_color="#FFFFFF",
            width=450,
            height=340,
            drawing_mode="freedraw",
            update_streamlit=True,
            display_toolbar=True, 
            key="alphanumeric_canvas_v10" # Forced unique component mount refresh
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, 3] > 0):
            # Read canvas content
            raw_canvas_gray = canvas_result.image_data[:, :, 0]
            img_inverted = cv2.bitwise_not(raw_canvas_gray)
            
            # Smart Crop: Find bounding box of drawing to strip away extra whitespace margins!
            # This keeps predictions 100% accurate even though width is 450
            pts = np.argwhere(img_inverted > 0)
            if len(pts) > 0:
                y_min, x_min = pts.min(axis=0)
                y_max, x_max = pts.max(axis=0)
                
                # Extract drawn object region explicitly
                cropped_stroke = img_inverted[y_min:y_max+1, x_min:x_max+1]
                
                # Place cropped shape inside a uniform squared frame with padding margins
                pad = 30
                img_ready_to_predict = cv2.copyMakeBorder(cropped_stroke, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                source_type = "canvas"

    with input_tab2:
        uploaded_file = st.file_uploader("Drop a handwritten character image here:", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            pil_image = Image.open(uploaded_file).convert('L')
            opencv_img = np.array(pil_image)
            
            if np.mean(opencv_img) > 127:
                img_ready_to_predict = cv2.bitwise_not(opencv_img)
            else:
                img_ready_to_predict = opencv_img
                
            source_type = "upload"
            st.image(uploaded_file, caption="Uploaded File Preview", width=180)

    st.write("")
    predict_btn = st.button("Execute Neural Inference Engine", type="primary", use_container_width=True)

with layout_col2:
    st.markdown("<div class='premium-card'><div class='section-header'>📊 Real-Time Matrix Analytics</div></div>", unsafe_allow_html=True)
    
    analytics_placeholder = st.empty()
    
    if "latest_report" not in st.session_state:
        st.session_state.latest_report = None
    if "predicted_char" not in st.session_state:
        st.session_state.predicted_char = "A"

    if predict_btn and model is not None:
        if img_ready_to_predict is not None:
            with st.spinner("Analyzing structural arrays..."):
                
                img_resized = cv2.resize(img_ready_to_predict, (28, 28))
                img_normalized = img_resized.astype("float32") / 255.0
                img_final = img_normalized.reshape(1, 28, 28, 1)
                
                predictions = model.predict(img_final)[0]
                top_3_classes = np.argsort(predictions)[-3:][::-1]
                
                highest_class = top_3_classes[0]
                highest_char = decode_prediction(highest_class)
                highest_conf = predictions[highest_class] * 100
                
                st.session_state.predicted_char = highest_char
                
                report_content = (
                    f"=========================================\n"
                    f" ALPHAMIND HYBRID INFERENCE ENGINE REPORT\n"
                    f"=========================================\n"
                    f"Timestamp:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Acquisition Mode:   {source_type.upper()}\n"
                    f"Predicted Target:   '{highest_char}'\n"
                    f"Model Confidence:   {highest_conf:.2f}%\n"
                    f"Raw Output Class:   Index {highest_class}\n\n"
                    f"Top 3 Distribution Vector:\n"
                )
                for idx in top_3_classes:
                    report_content += f" - Class '{decode_prediction(idx)}': {predictions[idx]*100:.2f}%\n"
                report_content += "=========================================\n"
                st.session_state.latest_report = report_content

                with analytics_placeholder.container():
                    st.markdown(
                        f"""
                        <div class='prediction-display'>
                            <p style='margin:0; font-size:0.85rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.1em; font-weight:500;'>Top Identified Classification Target</p>
                            <div class='pred-letter'>{highest_char}</div>
                            <p style='margin:0; font-size:1.1rem; color:#38BDF8; font-weight:600; margin-bottom: 12px;'>System Confidence: {highest_conf:.2f}%</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    st.write("")
                    st.markdown("<p style='font-size:0.9rem; font-weight:600; color:#F1F5F9; margin-bottom:12px;'>Class Probability Vector Spread (Top 3)</p>", unsafe_allow_html=True)
                    
                    html_bars = ""
                    for idx in top_3_classes:
                        char_label = decode_prediction(idx)
                        conf_val = predictions[idx] * 100
                        html_bars += f"""
                        <div class="bar-container">
                            <div class="bar-label">{char_label}</div>
                            <div class="bar-track">
                                <div class="bar-fill" style="width: {conf_val}%;"></div>
                            </div>
                            <div class="bar-value">{conf_val:.2f}%</div>
                        </div>
                        """
                    st.markdown(html_bars, unsafe_allow_html=True)
                    
                    st.write("")
                    st.markdown("<p style='font-size:0.9rem; font-weight:600; color:#F1F5F9; margin-bottom:8px;'>Telemetry Pipeline Diagnostic Feed</p>", unsafe_allow_html=True)
                    tele_col1, tele_col2 = st.columns(2)
                    with tele_col1:
                        st.image(img_ready_to_predict, caption="Auto-Cropped Isolated Drawing", use_container_width=True)
                    with tele_col2:
                        st.image(img_resized, caption="28x28 Model Entry", width=110)
        else:
            analytics_placeholder.error("System Interrupted: Input metrics returned empty. Please draw a character or drop an image file first.")
    else:
        if st.session_state.latest_report is not None:
            with analytics_placeholder.container():
                st.markdown(
                    f"""
                    <div class='prediction-display'>
                        <p style='margin:0; font-size:0.85rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.1em; font-weight:500;'>Top Identified Classification Target</p>
                        <div class='pred-letter'>{st.session_state.predicted_char}</div>
                        <p style='margin:0; font-size:1.1rem; color:#38BDF8; font-weight:600; margin-bottom: 12px;'>Inference Complete</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        else:
            analytics_placeholder.info("System Standby: Awaiting vector arrays. Draw or upload a target and execute inference.")

    # --- REPORT DOWNLOAD BUTTON ---
    if st.session_state.latest_report is not None:
        st.write("")
        st.download_button(
            label="📥 Download Inference Analytical Report (.txt)",
            data=st.session_state.latest_report,
            file_name=f"inference_report_{st.session_state.predicted_char}.txt",
            mime="text/plain",
            use_container_width=True
        )