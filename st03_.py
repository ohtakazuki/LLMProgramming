# 必要なモジュールをインポート
import os
from dotenv import load_dotenv
import streamlit as st
from langchain import PromptTemplate, OpenAI, LLMChain

# 環境変数の読み込み
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.environ['API_KEY']

# 言語モデルChainの作成
llm = OpenAI(max_tokens=300, temperature=1.2)

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template("{question}"),
)

# 質問の入力
st.text_input("質問を入力してください", key="question")

# 質問の取得
question = st.session_state.question

if question!="":
    # 言語モデルからの回答を表示
    st.write(llm_chain.run(question))
