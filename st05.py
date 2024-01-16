# 必要なモジュールをインポート
import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

# 環境変数の取得
load_dotenv()
# OpenAI APIクライアントを生成
client = OpenAI(api_key=os.environ['API_KEY'])

# タイトル
st.title("💬 Chatbot")

# メッセージリストが空なら初期メッセージを設定
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "ようこそ！ご要件は何ですか？"}]

# メッセージリストの内容をチャット要素に表示
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# チャット入力をpromptに代入かつ判定
if prompt := st.chat_input():
    # ユーザーからの質問をメッセージリストに追加
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ユーザーからの質問をチャット要素に表示
    st.chat_message("user").write(prompt)

    # 言語モデルへリクエスト
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=st.session_state.messages,
        stream=True)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_message = ""

        # 言語モデルからの回答を取得
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
              full_message += chunk.choices[0].delta.content
            message_placeholder.markdown(full_message + "▌")

        message_placeholder.markdown(full_message)

        # 言語モデルからの回答をメッセージリストに追加
        st.session_state.messages.append({"role": "assistant", "content": full_message})