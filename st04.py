import streamlit as st

value = 0

if st.button('＋'):
    value += 1

if st.button('ー'):
    value -= 1

f"値：{value}"
