# =========================
# Liver Cancer MRI Classification Web App
# DenseNet121 + Grad-CAM
# Colorful Dynamic Final UI
# =========================

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.cm as cm
from PIL import Image
from tensorflow.keras.models import load_model

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Liver Cancer MRI Classifier",
    page_icon="🩺",
    layout="wide"
)

# -------------------------
# Custom CSS Styling
# -------------------------
st.markdown("""
    <style>
    /* ==============================
       GLOBAL APP BACKGROUND
    ============================== */
    .stApp {
        background: linear-gradient(-45deg, #0f172a, #111827, #1e293b, #0f766e);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* ==============================
       SIDEBAR
    ============================== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15,118,110,0.95), rgba(8,145,178,0.95), rgba(29,78,216,0.95));
        backdrop-filter: blur(18px);
        border-right: 1px solid rgba(255,255,255,0.12);
        box-shadow: 4px 0px 25px rgba(0,0,0,0.25);
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    .sidebar-box {
        background: rgba(255, 255, 255, 0.10);
        padding: 18px;
        border-radius: 18px;
        margin-bottom: 20px;
        box-shadow: 0px 8px 25px rgba(0,0,0,0.22);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.14);
        transition: all 0.3s ease-in-out;
    }

    .sidebar-box:hover {
        transform: translateY(-3px);
        box-shadow: 0px 10px 28px rgba(0,0,0,0.35);
    }

    .sidebar-title {
        font-size: 22px;
        font-weight: 800;
        margin-bottom: 10px;
        color: #ffffff;
        text-shadow: 0px 0px 8px rgba(255,255,255,0.25);
    }

    .sidebar-text {
        font-size: 15px;
        color: #f8fafc;
        line-height: 1.8;
    }

    /* ==============================
       TITLES
    ============================== */
    .main-title {
        font-size: 48px;
        font-weight: 900;
        text-align: center;
        color: #38bdf8;
        margin-bottom: 8px;
        text-shadow: 0px 0px 18px rgba(56,189,248,0.55);
        letter-spacing: 1px;
    }

    .sub-title {
        font-size: 19px;
        text-align: center;
        color: #e2e8f0;
        margin-bottom: 30px;
        font-weight: 500;
    }

    /* ==============================
       GLASS CARD
    ============================== */
    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        padding: 24px;
        border-radius: 22px;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.35);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.14);
        margin-bottom: 22px;
        transition: all 0.35s ease;
    }

    .glass-card:hover {
        transform: translateY(-5px) scale(1.01);
        box-shadow: 0px 14px 35px rgba(0,0,0,0.42);
        border: 1px solid rgba(56,189,248,0.35);
    }

    /* ==============================
       RESULT CARD
    ============================== */
    .result-card {
        background: linear-gradient(135deg, rgba(14,165,233,0.95), rgba(37,99,235,0.95), rgba(59,130,246,0.95));
        padding: 24px;
        border-radius: 22px;
        box-shadow: 0px 12px 30px rgba(37,99,235,0.45);
        color: white;
        text-align: center;
        margin-bottom: 22px;
        border: 1px solid rgba(255,255,255,0.15);
        backdrop-filter: blur(12px);
        animation: pulseGlow 2.8s infinite alternate;
    }

    @keyframes pulseGlow {
        from { box-shadow: 0px 12px 30px rgba(37,99,235,0.35); }
        to { box-shadow: 0px 14px 36px rgba(56,189,248,0.65); }
    }

    .prediction-text {
        font-size: 30px;
        font-weight: 900;
        color: #ffffff;
        text-shadow: 0px 0px 12px rgba(255,255,255,0.25);
    }

    .confidence-text {
        font-size: 23px;
        font-weight: 700;
        color: #e0f2fe;
        margin-top: 8px;
    }

    /* ==============================
       SECTION TITLE
    ============================== */
    .section-title {
        font-size: 28px;
        font-weight: 800;
        color: #22d3ee;
        margin-top: 22px;
        margin-bottom: 16px;
        text-shadow: 0px 0px 10px rgba(34,211,238,0.28);
    }

    /* ==============================
       TEXT / NOTES
    ============================== */
    .small-note {
        font-size: 15px;
        color: #e2e8f0;
        line-height: 1.8;
    }

    .footer {
        text-align: center;
        font-size: 14px;
        color: #cbd5e1;
        margin-top: 45px;
        padding: 16px;
        background: rgba(255,255,255,0.05);
        border-radius: 16px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* ==============================
       BUTTONS
    ============================== */
    .stButton > button {
        width: 100%;
        border-radius: 14px;
        padding: 0.75rem 1rem;
        font-size: 17px;
        font-weight: 700;
        border: none;
        color: white;
        background: linear-gradient(135deg, #06b6d4, #2563eb);
        box-shadow: 0px 8px 22px rgba(37,99,235,0.35);
        transition: all 0.3s ease-in-out;
    }

    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0px 12px 28px rgba(56,189,248,0.45);
        background: linear-gradient(135deg, #0ea5e9, #1d4ed8);
    }

    /* ==============================
       FILE UPLOADER
    ============================== */
    .stFileUploader {
        background: rgba(255,255,255,0.06);
        padding: 18px;
        border-radius: 18px;
        border: 1px dashed rgba(56,189,248,0.4);
        backdrop-filter: blur(10px);
        box-shadow: 0px 6px 20px rgba(0,0,0,0.2);
    }

    /* ==============================
       PROGRESS BAR
    ============================== */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #22d3ee, #3b82f6, #06b6d4);
        border-radius: 10px;
    }

    /* ==============================
       METRIC BOXES
    ============================== */
    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        padding: 14px;
        border-radius: 18px;
        backdrop-filter: blur(10px);
        box-shadow: 0px 6px 18px rgba(0,0,0,0.22);
    }

    /* ==============================
       IMAGE STYLING
    ============================== */
    img {
        border-radius: 18px;
        box-shadow: 0px 8px 24px rgba(0,0,0,0.28);
    }

    /* ==============================
       HIDE STREAMLIT DEFAULTS
    ============================== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------
st.markdown('<div class="main-title">🩺 Liver Cancer MRI Classification</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">DenseNet121-Based Multiclass Liver Cancer Detection with Explainable AI (Grad-CAM)</div>',
    unsafe_allow_html=True
)

# -------------------------
# Constants
# -------------------------
IMG_HEIGHT = 224
IMG_WIDTH = 224

CLASS_NAMES = [
    "Angiosarcoma",
    "Cholangiocarcinoma",
    "Healthy",
    "Hemangioma",
    "Hepatocellular_Carcinoma"
]

MODEL_PATH = "final_liver_densenet121_model.keras"
LAST_CONV_LAYER_NAME = "conv5_block16_concat"

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_trained_model():
    model = load_model(MODEL_PATH)
    return model

model = load_trained_model()

# -------------------------
# Preprocess Image
# -------------------------
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img).astype("float32") / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img, img_array, img_batch

# -------------------------
# Predict Class
# -------------------------
def predict_image(model, img_batch):
    preds = model.predict(img_batch, verbose=0)

    if isinstance(preds, list):
        preds = preds[0]

    preds = np.array(preds)

    predicted_index = int(np.argmax(preds[0]))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(np.max(preds[0])) * 100

    return predicted_class, confidence, preds[0], predicted_index

# -------------------------
# Generate Grad-CAM
# -------------------------
def generate_gradcam(model, img_batch, predicted_index):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(LAST_CONV_LAYER_NAME).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch, training=False)

        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        predictions = tf.convert_to_tensor(predictions)
        class_channel = predictions[:, predicted_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    return heatmap

# -------------------------
# Overlay Heatmap
# -------------------------
def overlay_gradcam(original_img_array, heatmap):
    original_img = np.uint8(original_img_array * 255)

    heatmap_resized = cv2.resize(heatmap, (IMG_WIDTH, IMG_HEIGHT))
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
    heatmap_colored = np.uint8(255 * heatmap_colored)

    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)

    return heatmap_resized, superimposed_img

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.markdown("""
        <div class="sidebar-box">
            <div class="sidebar-title">⚙️ Project Info</div>
            <div class="sidebar-text">
                <b>Model:</b> DenseNet121<br>
                <b>Task:</b> Multiclass Liver MRI Classification<br>
                <b>Classes:</b> 5<br>
                <b>Explainability:</b> Grad-CAM
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="sidebar-box">
            <div class="sidebar-title">📌 Class Labels</div>
            <div class="sidebar-text">
                • Angiosarcoma<br>
                • Cholangiocarcinoma<br>
                • Healthy<br>
                • Hemangioma<br>
                • Hepatocellular Carcinoma
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="sidebar-box">
            <div class="sidebar-title">🧠 AI Features</div>
            <div class="sidebar-text">
                • MRI Image Upload<br>
                • Cancer Class Prediction<br>
                • Confidence Score<br>
                • Class Probability Analysis<br>
                • Explainable AI Visualization
            </div>
        </div>
    """, unsafe_allow_html=True)

# -------------------------
# Upload Section
# -------------------------
st.markdown('<div class="section-title">📤 Upload MRI Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    img_pil, img_array, img_batch = preprocess_image(uploaded_file)

    with st.spinner("🔍 Analyzing MRI image... Please wait"):
        predicted_class, confidence, probs, predicted_index = predict_image(model, img_batch)
        heatmap = generate_gradcam(model, img_batch, predicted_index)
        heatmap_resized, gradcam_overlay = overlay_gradcam(img_array, heatmap)

    # -------------------------
    # Row 1: Image + Prediction
    # -------------------------
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🖼 Uploaded MRI Image")
        st.image(img_pil, width=250)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="result-card">
                <div style="font-size:20px;">🧠 Prediction Result</div>
                <div class="prediction-text">{predicted_class}</div>
                <div class="confidence-text">Confidence: {confidence:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Confidence Meter")
        st.progress(min(int(confidence), 100))
        st.markdown(f"<p class='small-note'>The model is highly confident about the predicted class.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------
    # Probabilities
    # -------------------------
    st.markdown('<div class="section-title">📈 Class Probability Distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    for i, class_name in enumerate(CLASS_NAMES):
        prob = float(probs[i]) * 100
        st.write(f"**{class_name}** — {prob:.2f}%")
        st.progress(min(int(prob), 100))

    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------
    # Grad-CAM Section
    # -------------------------
    st.markdown('<div class="section-title">🧠 Explainable AI (Grad-CAM)</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🔥 Grad-CAM Heatmap")
        st.image(heatmap_resized, width=250, clamp=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🩻 Heatmap Overlay on MRI")
        st.image(gradcam_overlay, width=250)
        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------
    # Interpretation
    # -------------------------
    st.markdown('<div class="section-title">📘 Interpretation</div>', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="glass-card">
            <p class="small-note">
                <b>Predicted Class:</b> {predicted_class}<br><br>
                <b>Confidence:</b> {confidence:.2f}%<br><br>
                The Grad-CAM visualization highlights the MRI regions that contributed most strongly to the model’s prediction.
                Warmer areas indicate stronger influence on the DenseNet121 classification decision.
            </p>
        </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
        <div class="glass-card">
            <h3 style="color:#67e8f9;">👋 Welcome</h3>
            <p class="small-note">
                Upload a liver MRI image to classify it into one of the following categories:
                <br><br>
                • Angiosarcoma<br>
                • Cholangiocarcinoma<br>
                • Healthy<br>
                • Hemangioma<br>
                • Hepatocellular Carcinoma
                <br><br>
                The app will also generate a <b>Grad-CAM heatmap</b> to explain which regions influenced the prediction.
            </p>
        </div>
    """, unsafe_allow_html=True)

# -------------------------
# Footer
# -------------------------
st.markdown("""
    <div class="footer">
        Developed for Academic Deep Learning Project | DenseNet121 + MRI + Explainable AI
    </div>
""", unsafe_allow_html=True)