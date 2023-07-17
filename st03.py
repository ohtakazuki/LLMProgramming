import streamlit as st

# session_stateにvalueが無ければ0で初期化
if 'value' not in st.session_state: 
    st.session_state.value = 0

# 変数にsession_stateから値を取得
value = st.session_state.value

# ＋ボタンを定義：1加算
if st.button('＋'):
    value += 1

# ーボタンを定義：1減算
if st.button('ー'):
    value -= 1

# 変数の値を表示
f"値：{value}"

# session_stateに変数の値を保存
st.session_state.value = value
