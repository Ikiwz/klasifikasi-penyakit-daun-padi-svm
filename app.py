import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# Fungsi untuk memuat model (dengan cache agar tidak loading berulang)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model_padi.h5')
    return model


model = load_model()

# Judul Aplikasi
st.title("Klasifikasi Penyakit Daun Padi 🌾")
st.write("Unggah gambar daun padi untuk mengetahui kondisinya (Blast, Blight, Tungro, atau Normal).")

# Widget untuk upload file
uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah.', use_column_width=True)
    st.write("")
    st.write("Menganalisis...")

    # Pra-pemrosesan gambar
    img_array = np.array(image.resize((150, 150)))  # Ubah ukuran ke 150x150
    img_array = img_array / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Tambah dimensi batch

    # Lakukan prediksi
    prediction = model.predict(img_array)
    class_names = ['Blast', 'Blight', 'Normal', 'Tungro']  # Sesuaikan urutan dengan class_indices

    # Dapatkan hasil prediksi
    score = tf.nn.softmax(prediction[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    # Tampilkan hasil
    st.subheader(f"Hasil Prediksi: {predicted_class}")
    st.write(f"Tingkat Keyakinan: {confidence:.2f}%")