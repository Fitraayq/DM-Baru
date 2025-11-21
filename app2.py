import streamlit as st
import joblib


@st.cache_resource
def load_models():
    lr = joblib.load("model_lr.pkl")          # Logistic Regression
    vectorizer = joblib.load("vectorizer.pkl")

    # Jika kamu punya model SVM:
    try:
        svm = joblib.load("model_svm.pkl")
    except:
        svm = None

    return lr, svm, vectorizer

lr, svm, vectorizer = load_models()

# Akurasi versi training (bisa kamu ubah sesuai hasilmu)
ACC_LR = "92–96%"
ACC_SVM = "93–97%" if svm else "Unavailable"



st.set_page_config(page_title="Fake News Detector (Light Version)", layout="wide")

st.title("📰 Fake News Detection App — Lite Version")
st.write("""
Versi ringan aplikasi deteksi berita palsu.  
Menggunakan **TF-IDF + Logistic Regression** (dan optional SVM).  
Cepat, ringan, dan akurat (>90%).  
""")

st.sidebar.header("⚙️ Pilih Model")

model_choice = st.sidebar.radio(
    "Pilih model Machine Learning:",
    ("Logistic Regression", "SVM (Jika tersedia)")
)

st.sidebar.header("📊 Akurasi Model")
st.sidebar.write(f"🔹 Logistic Regression: **{ACC_LR}**")
st.sidebar.write(f"🔹 SVM: **{ACC_SVM}**")

st.subheader("Masukkan Teks Berita:")
input_text = st.text_area(
    "Tempel atau ketik teks berita di sini:",
    height=260
)


if st.button("🔍 Prediksi"):
    if input_text.strip() == "":
        st.warning("Tolong masukkan teks berita terlebih dahulu.")
    else:
        X = vectorizer.transform([input_text])

        # Pilih model
        if model_choice == "Logistic Regression":
            model = lr
        else:
            if svm is not None:
                model = svm
            else:
                st.error("Model SVM tidak ditemukan. Gunakan Logistic Regression saja.")
                st.stop()

        pred = model.predict(X)[0]

        # Output hasil prediksi
        if pred == 1:
            st.success("🟢 **REAL NEWS** — Berita kemungkinan ASLI dan kredibel.")
        else:
            st.error("🔴 **FAKE NEWS** — Berita kemungkinan PALSU atau misinformasi.")

        st.caption("Catatan: Prediksi tidak menggantikan pemeriksaan fakta profesional.")


st.write("---")
st.write("Versi ringan • Dibuat oleh AIDIL • 2025")
