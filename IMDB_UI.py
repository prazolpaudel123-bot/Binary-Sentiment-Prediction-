import streamlit as st
import pickle
import numpy as np
import os
import re
import string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Load Resources
# -----------------------------
@st.cache_resource
def load_resources():
    model_path = r"L:\Softwares\VS Code\Files\RNN\imdb_binary_sentiment\imdb_sentiment_simplernn.h5"
    tokenizer_path = r"L:\Softwares\VS Code\Files\RNN\imdb_binary_sentiment\tokenizer_imdb_sentiment.pickle"
    max_length_path = r"L:\Softwares\VS Code\Files\RNN\imdb_binary_sentiment\max_length_imdb.pickle"

    # Check files
    missing = []
    for path in [model_path, tokenizer_path, max_length_path]:
        if not os.path.exists(path):
            missing.append(path)

    if missing:
        st.error("❌ Missing files:")
        for m in missing:
            st.write(f"- `{m}`")
        st.info("Please put the model, tokenizer and max_length files in the same folder as app.py")
        st.stop()

    model = load_model(model_path)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    with open(max_length_path, "rb") as f:
        max_length = pickle.load(f)

    return model, tokenizer, max_length

model, tokenizer, max_length = load_resources()

# -----------------------------
# Text Cleaning (same as training)
# -----------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# Prediction Function
# -----------------------------
def predict_sentiment(text: str):
    if not text.strip():
        return None, None, None

    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=max_length, padding="post", truncating="post")

    pred_prob = model.predict(padded, verbose=0)[0][0]

    if pred_prob >= 0.5:
        label = "Positive 😊"
        confidence = pred_prob
    else:
        label = "Negative 😞"
        confidence = 1 - pred_prob

    return label, confidence, pred_prob

# -----------------------------
# UI
# -----------------------------
st.title("🎬 IMDb Plot Sentiment Analyzer")
st.markdown("Paste a movie/TV show **plot description** and the model will predict whether it sounds **Positive** or **Negative**.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Info")
    st.write(f"- Model: **SimpleRNN**")
    st.write(f"- Max sequence length: `{max_length}`")
    st.write(f"- Vocabulary size: `{len(tokenizer.word_index)}`")
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("""
    The model was trained on plot descriptions from the IMDb dataset.  
    Rating ≥ 7.0 → Positive  
    Rating < 7.0 → Negative
    """)
    st.markdown("---")
    st.markdown("Made with Streamlit + TensorFlow")

# Main input
description = st.text_area(
    "Enter plot description:",
    height=180,
    placeholder="Example: A high school chemistry teacher diagnosed with inoperable lung cancer turns to manufacturing and selling methamphetamine in order to secure his family's future."
)

# Example buttons
st.markdown("#### 💡 Try these examples:")
col1, col2, col3 = st.columns(3)

example_pos = "Decades after their 1984 All Valley Karate Tournament bout, a middle-aged Daniel LaRusso and Johnny Lawrence find themselves martial-arts rivals again."
example_neg = "In this high-octane action thriller, a cross-country road trip becomes a highway to hell for Brenda and her family. Alone in the New Mexico desert, they have to fight for their lives."
example_neutral = "A young woman takes a trip to romantic Verona, Italy, after a breakup, only to find that the villa she reserved was double-booked."

if col1.button("Positive Example"):
    description = example_pos
if col2.button("Negative Example"):
    description = example_neg
if col3.button("Another Example"):
    description = example_neutral

# Predict button
if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
    if not description.strip():
        st.warning("Please enter a plot description first.")
    else:
        with st.spinner("Analyzing..."):
            label, confidence, raw_score = predict_sentiment(description)

        st.markdown("---")
        st.subheader("Prediction Result")

        if "Positive" in label:
            st.success(f"### {label}")
        else:
            st.error(f"### {label}")

        # Confidence
        st.metric("Confidence", f"{confidence*100:.1f}%")
        st.progress(float(confidence))

        # Raw score
        st.caption(f"Raw model score: `{raw_score:.4f}`  (closer to 1.0 = more positive)")

        # Interpretation
        if raw_score >= 0.7:
            st.info("The model is quite confident this plot sounds **positive**.")
        elif raw_score <= 0.3:
            st.info("The model is quite confident this plot sounds **negative**.")
        else:
            st.info("The model is somewhat uncertain (score close to 0.5).")

# Footer
st.markdown("---")
st.caption("Binary Sentiment Classification | SimpleRNN | Trained on IMDb plot descriptions")