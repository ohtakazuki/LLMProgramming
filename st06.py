# 必要なモジュールをインポート
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
import streamlit as st

# 環境変数の読み込み
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ["API_KEY"]

# Retrieverの作成
def create_retriever():
    # PDFの読込
    loader = PyPDFLoader("./data2/001615363.pdf")
    documents = loader.load()

    # チャンクに分割
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=512, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Indexの構築
    db = Chroma.from_documents(texts, OpenAIEmbeddings())

    # 検索（Retriever）の取得
    retriever = db.as_retriever()

    st.session_state["retriever"] = retriever

    return retriever

# AgentExecutorの作成
def create_agent_executor():
    tools = [
        create_retriever_tool(
            retriever,
            "ForeignTravelersQASystem",
            "訪日外国人の旅行や土産品、消費動向に関する質問に役立ちます。",
        )
    ]

    # モデルの作成
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=1.2)

    prompt = hub.pull("hwchase17/openai-functions-agent")

    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    st.session_state["agent_executor"] = agent_executor

    return agent_executor


# タイトル
st.title("Index検索")

# ステータス表示用
status = st.empty()

# Indexの準備
if "retriever" not in st.session_state:
    status.markdown("Index構築中")
    retriever = create_retriever()
else:
    retriever = st.session_state["retriever"]

# Agentの準備
if "agent_executor" not in st.session_state:
    status.markdown("Agent準備中")
    agent_executor = create_agent_executor()
else:
    agent_executor = st.session_state["agent_executor"]

status.markdown("Index/Agent準備OK")

# 質問入力欄
query = st.text_input("質問：")

# Submit時の処理
if query!="":
    status.markdown("回答生成中")
    agent_executor = create_agent_executor()
    result = agent_executor.invoke({"input": query})
    result["output"]
    status.markdown("Index/Agent準備OK")
