import os
from flask import Flask, g, request, render_template
from dotenv import load_dotenv
import tiktoken
from llama_index.core import StorageContext, load_index_from_storage, ChatPromptTemplate, Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.text_splitter import SentenceSplitter

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ["API_KEY"]

app = Flask(__name__)

PERSIST_DIR = "./storage02"
MODEL_NAME = "gpt-3.5-turbo-0125"

def get_text_qa_template():
    """テキストQAテンプレートを返す"""
    system_prompt = ChatMessage(
        content=(
            "あなたは世界中で信頼されているQAシステムです。"
            "事前知識ではなく、常に提供されたコンテキスト情報を使用してクエリに回答してください。"
            "従うべきいくつかのルール:"
            "1. 回答内で指定されたコンテキストを直接参照しないでください。"
            "2. 「コンテキストに基づいて、...」や「コンテキスト情報は...」、またはそれに類するような記述は避けてください。"
        ),
        role=MessageRole.SYSTEM,
    )

    message_templates = [
        system_prompt,
        ChatMessage(
            content=(
                "コンテキスト情報は以下のとおりです。"
                "---------------------"
                "{context_str}"
                "---------------------"
                "事前知識ではなくコンテキスト情報を考慮して、クエリに答えます。"
                "Query: {query_str}"
                "Answer: "
            ),
            role=MessageRole.USER,
        ),
    ]

    return ChatPromptTemplate(message_templates=message_templates)

def get_refine_template():
    """リファインテンプレートを返す"""
    refine_message_templates = [
        ChatMessage(
            content=(
                "あなたは、既存の回答を改良する際に2つのモードで厳密に動作するQAシステムのエキスパートです。"
                "1. 新しいコンテキストを使用して元の回答を**書き直す**。"
                "2. 新しいコンテキストが役に立たない場合は、元の回答を**繰り返す**。"
                "回答内で元の回答やコンテキストを直接参照しないでください。"
                "疑問がある場合は、元の答えを繰り返してください。"
                "New Context: {context_msg}"
                "Query: {query_str}"
                "Original Answer: {existing_answer}"
                "New Answer: "
            ),
            role=MessageRole.USER,
        )
    ]

    return ChatPromptTemplate(message_templates=refine_message_templates)

def get_llama_index():
    """Llama_Indexを返す"""
    if "llama_index" not in g:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        g.llama_index = load_index_from_storage(storage_context)
    return g.llama_index

def get_query_engine():
    """Query Engineを返す"""
    llama_index = get_llama_index()

    node_parser = SentenceSplitter(
        separator="。",
        chunk_size=1024,
        chunk_overlap=20,
        tokenizer=tiktoken.encoding_for_model(MODEL_NAME).encode
    )

    llm = OpenAI(model=MODEL_NAME, temperature=1.2)

    Settings.llm = llm
    Settings.node_parser = node_parser

    return llama_index.as_query_engine(
        similarity_top_k=3,
        streaming=True,
        text_qa_template=get_text_qa_template(),
        refine_template=get_refine_template(),
    )

def process_query(question):
    """質問を処理し、回答と情報源を返す"""
    query_engine = get_query_engine()
    response = query_engine.query(question)
    answer = str(response)
    source = response.get_formatted_sources()
    return answer, source

@app.route("/", methods=["GET", "POST"])
def index():
    """リクエストを処理し、回答を表示"""
    if request.method == "POST":
        question = request.form["question"]
        answer, source = process_query(question)
    else:
        question = answer = source = ""

    return render_template("index.html", response=answer, question=question, source=source)

if __name__ == "__main__":
    app.run(port=5000)