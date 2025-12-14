import streamlit as st
from transformers import GPT2LMHeadModel, AutoTokenizer
import torch

# ============================================================
# LOAD MODEL & TOKENIZER
# ============================================================

MODEL_PATH = "./indo-scriptgen"   # folder model setelah training selesai

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return tokenizer, model, device

tokenizer, model, device = load_model()

# ============================================================
# STREAMLIT UI
# ============================================================

st.title("🎬 Script Video Generator Bahasa Indonesia")
st.write("Masukkan **judul atau tema**, lalu model akan membuatkan script videonya.")

judul = st.text_input("Masukkan Judul / Tema Video:", "")

max_len = st.slider(
    "Panjang Maksimal Output (token)",
    min_value=100, max_value=700, value=300,
)

temperature = st.slider(
    "Creativity (temperature)",
    min_value=0.1, max_value=1.5, value=0.8, step=0.1
)

top_p = st.slider(
    "Top-p Sampling",
    min_value=0.1, max_value=1.0, value=0.9, step=0.05
)

generate_btn = st.button("🚀 Generate Script")

# ============================================================
# GENERATION LOGIC
# ============================================================

if generate_btn:
    if judul.strip() == "":
        st.warning("Masukkan judul atau tema terlebih dahulu!")
    else:
        with st.spinner("Sedang membuat script... tunggu sebentar..."):
            prompt = f"Judul: {judul}\nScript:"

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            output = model.generate(
                **inputs,
                max_length=max_len,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id
            )

            decoded = tokenizer.decode(output[0], skip_special_tokens=True)

            # Hapus prompt supaya tampil script bersih
            if decoded.startswith(prompt):
                generated_script = decoded[len(prompt):].strip()
            else:
                generated_script = decoded

        st.subheader("📄 Script Video:")
        st.write(generated_script)

        # Optional: untuk copy/paste
        st.text_area("Copy Script:", generated_script, height=250)
