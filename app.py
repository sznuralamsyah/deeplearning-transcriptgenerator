import streamlit as st
import os

# ============================================================
# ENV FIX (WAJIB UNTUK STREAMLIT CLOUD)
# ============================================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # paksa CPU

import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "./indo-scriptgen"
DEVICE = "cpu"

# ============================================================
# LOAD MODEL & TOKENIZER (CACHE)
# ============================================================
@st.cache_resource(show_spinner="Loading model, please wait...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = GPT2LMHeadModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,   # WAJIB CPU
        low_cpu_mem_usage=True
    )

    model.to(DEVICE)
    model.eval()

    return tokenizer, model

tokenizer, model = load_model()

# ============================================================
# STREAMLIT UI
# ============================================================
st.title("🎬 Script Video Generator (Bahasa Indonesia)")
st.caption("Powered by Indo GPT-2 (CPU only)")

st.markdown(
    "Masukkan **judul atau tema video**, lalu AI akan membuatkan "
    "script video programming dalam bahasa Indonesia."
)

judul = st.text_input(
    "Judul / Tema Video",
    placeholder="Contoh: Cara kerja REST API untuk pemula"
)

col1, col2, col3 = st.columns(3)

with col1:
    max_len = st.slider(
        "Max Output Token",
        min_value=150,
        max_value=600,
        value=300,
        step=50
    )

with col2:
    temperature = st.slider(
        "Creativity",
        min_value=0.3,
        max_value=1.3,
        value=0.8,
        step=0.1
    )

with col3:
    top_p = st.slider(
        "Top-p",
        min_value=0.5,
        max_value=1.0,
        value=0.9,
        step=0.05
    )

generate_btn = st.button("🚀 Generate Script", use_container_width=True)

# ============================================================
# GENERATION
# ============================================================
if generate_btn:
    if not judul.strip():
        st.warning("⚠️ Masukkan judul atau tema terlebih dahulu.")
    else:
        with st.spinner("✍️ Sedang membuat script..."):
            prompt = (
                "Buatkan script video programming berbahasa Indonesia.\n"
                f"Judul: {judul}\n"
                "Script:\n"
            )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=128
            )

            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_length=max_len,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id
                )

            decoded = tokenizer.decode(
                output[0],
                skip_special_tokens=True
            )

            # Bersihkan prompt
            generated_script = decoded.replace(prompt, "").strip()

        st.subheader("📄 Script Video")
        st.write(generated_script)

        st.text_area(
            "📋 Copy Script",
            generated_script,
            height=300
        )
