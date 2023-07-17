import streamlit as st

# st.text_input("名前を入力してください", key="name")
# name = st.session_state.name

name = st.text_input("名前を入力してください")

if name!="":
    f"こんにちは！{name}さん"

"st.session_state.name:" + st.session_state.name