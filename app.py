import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from disease_info import disease_info

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="CropGuard 🌱", page_icon="🌿", layout="centered")

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1500382017468-9049fed747ef");
    background-size: cover;
    background-attachment: fixed;
}

/* Dark overlay */
.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    z-index: -1;
}

/* Title */
.title {
    text-align: center;
    font-size: 70px;
    color: #E8F5E9;
    font-weight: bold;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 28px;
    color: #C8E6C9;
    margin-bottom: 30px;
}

/* Upload (no box look) */
.upload-box {
    text-align: center;
    width: 70%;
    margin: auto;
    font-size: 22px;
    color: white;
}

/* File uploader */
.stFileUploader label {
    font-size: 22px !important;
    color: white !important;
}

/* Buttons */
button {
    font-size: 20px !important;
}

/* Glass card */
.card {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(10px);
    padding: 30px;
    border-radius: 20px;
    margin-top: 25px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.4);
    font-size: 22px;
    color: white;
}

/* Prediction text */
.card h2 {
    font-size: 36px;
    color: #A5D6A7;
}

.card h3 {
    font-size: 30px;
    color: #E8F5E9;
}

/* Section headings */
h3, .stSubheader {
    font-size: 26px !important;
    color: #C8E6C9 !important;
}

/* Normal text */
.stMarkdown p {
    font-size: 22px !important;
    color: #F1F8E9;
}

/* Progress */
.stProgress {
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
model = load_model("C:/Users/ritac/cropguard_model.keras")

# ⚠️ MUST MATCH EXACT ORDER
class_names = [
    'Banana_yellow_Sigatoka',
    'Tomato_BacterialL_spot',
    'banana_pseudostem_weevil_damage',
    'brinjal_Wet Rot',
    'cinnamon_StripeCanker',
    'healthy',
    'mango_Anthracnose',
    'mango_powder_mildew'
]

# -------------------- HEADER --------------------
st.markdown('<div class="title">🌱 CropGuard </div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart Plant Disease Detection & Recommendation System</div>', unsafe_allow_html=True)

# -------------------- UPLOAD --------------------
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload a plant image", type=["jpg", "png", "jpeg"])
st.markdown('</div>', unsafe_allow_html=True)

# -------------------- PROCESS --------------------
if uploaded_file is not None:

    # Show image
    img = image.load_img(uploaded_file, target_size=(224,224))
    st.image(img, caption="Uploaded Image", width=420)

    # Preprocess
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    pred = model.predict(img_array)
    predicted_class = class_names[np.argmax(pred)]
    confidence = np.max(pred) * 100

    # -------------------- RESULT --------------------
    st.markdown(f"""
    <div class="card">
        <h2>🌿 Prediction: {predicted_class}</h2>
        <h3>📊 Confidence: {confidence:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)

    st.progress(int(confidence))

    # -------------------- INFO --------------------
    info = disease_info.get(predicted_class)

    if info:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("🧠 Symptoms")
        for s in info["symptoms"]:
            st.write("✔️", s)

        st.subheader("🌿 Organic Treatment")
        for t in info["treatment"]:
            st.write("🌱", t)

        st.subheader("🛡️ Prevention")
        for p in info["prevention"]:
            st.write("✅", p)

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("No information found for this disease")

    # -------------------- WARNING --------------------
    if confidence < 50:
        st.warning("⚠️ Low confidence prediction. Please verify manually.")