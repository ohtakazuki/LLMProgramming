import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("./data2/iris.data", header=0)

# セレクトボックス
option = st.selectbox('品種：', set(df["Name"]))

# データフレームを抽出
df2 = df[df["Name"]==option]

# データフレームを表示
df2

# 散布図を表示
fig, ax = plt.subplots()
ax.scatter(df2["SepalLength"], df2["SepalWidth"])
st.pyplot(fig)