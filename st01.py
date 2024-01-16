import streamlit as st

name = st.text_input("名前を入力してください")

if name!="":
    f"こんにちは！{name}さん"