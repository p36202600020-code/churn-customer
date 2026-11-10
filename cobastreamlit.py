
import streamlit as st

st.title("Aplikasi Streamlit Pertama")
st.write("Halo, selamat datang di aplikasi saya!")

nama = st.text_input("Masukkan nama kamu")

if st.button("Kirim"):
    st.success(f"Halo, {nama}!")
