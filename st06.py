# 必要なモジュールをインポート
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.agents import initialize_agent
import streamlit as st

# 環境変数の読み込み
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ["API_KEY"]

# Indexの構築
def create_index():
    # PDFの読込
    loader = PyPDFLoader("./data2/001615363.pdf")
    documents = loader.load()

    # チャンクに分割
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=512, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Indexの構築
    db = Chroma.from_documents(texts, OpenAIEmbeddings())

    st.session_state["db"] = db

    return db

# Agentの作成
def create_agent():
    # RetrievalQAチェーンの作成
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=1.2),
        chain_type="stuff",
        retriever=db.as_retriever())

    tools = [
        Tool(
            name="訪日外国人旅行者QAシステム",
            func=qa.run,
            description="訪日外国人の旅行や土産品、消費動向に関する質問に役立ちます。",
        )]

    return initialize_agent(
        tools,
        llm=OpenAI(), 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,)

# タイトル
st.title("Index検索")

# ステータス表示用
status = st.empty()

# Indexの準備
if "db" not in st.session_state:
    status.markdown("Index構築中")
    db = create_index()
else:
    db = st.session_state["db"]

# Agentの準備
if "agent" not in st.session_state:
    status.markdown("Agent準備中")
    agent = create_agent()
else:
    agent = st.session_state["agent"]

status.markdown("Index/Agent準備OK")

# 質問入力欄
query = st.text_input("質問：")

# Submit時の処理
if query!="":
    status.markdown("回答生成中")
    agent = create_agent()
    result = agent.run(input=query)
    result
    status.markdown("Index/Agent準備OK")
