import streamlit as st

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis dan Clustering Pengguna Spotify",
    page_icon="🎵",
    layout="wide"
)

# Judul dan deskripsi
st.title("🎵 Analisis dan Clustering Pengguna Spotify")
st.markdown("""
Aplikasi ini melakukan analisis clustering pada data penggunaan Spotify untuk mengidentifikasi segmen pengguna 
dan menemukan pola dalam preferensi musik. 

**Status:** Aplikasi sedang dalam proses deployment. Silakan tunggu beberapa saat...
""")

# Cek environment dan secrets
if hasattr(st, 'secrets'):
    st.success("✅ Deployment berhasil! Secrets tersedia.")

    # Tampilkan informasi secrets (hanya untuk tujuan pengujian, hapus pada aplikasi produksi)
    if "some_section" in st.secrets:
        st.write("Nilai some_key:", st.secrets.some_section.some_key)
else:
    st.warning("⚠️ Aplikasi berjalan dalam mode lokal, secrets tidak tersedia.")

st.markdown("---")
st.markdown("Versi deployment uji - akan segera diganti dengan aplikasi lengkap")
