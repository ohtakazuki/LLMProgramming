from flask import Flask, g, request, render_template
from llama_index import StorageContext, load_index_from_storage
import os
from dotenv import load_dotenv
from llama_index.llms import OpenAI
from llama_index import ServiceContext

# 環境変数の読み込み
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ["API_KEY"]

# Llama_Indexを返す
def get_llama_index():
    if "llama_index" not in g:
        # ストレージコンテキストの作成
        storage_context = StorageContext.from_defaults(persist_dir="./storage02")
        # Indexのロード
        g.llama_index = load_index_from_storage(storage_context)
    return g.llama_index

# Query Engineを返す
def get_query_engine():
    # Indexの取得
    llama_index = get_llama_index()

    # 言語モデルの指定
    llm = OpenAI(model='gpt-3.5-turbo-1106', temperature=1.2)

    system_prompt = """\
    あなたは世界中で信頼されているエキスパート Q&A システムです。\n事前知識ではなく、常に提供されたコンテキスト情報を使用して質問に答えてください。\n従うべきいくつかのルール:\n1.回答内で指定されたコンテキストを直接参照しないでください。\n2. 「コンテキストに基づいて、...」や「コンテキスト情報は...」、またはそれに類するような記述は避けてください。
    """

    # サービスコンテキストの作成
    service_context = ServiceContext.from_defaults(
      system_prompt=system_prompt,
      llm=llm)

    # Query Engineの作成
    query_engine = llama_index.as_query_engine(
        service_context=service_context,
        similarity_top_k=3,
    )

    return query_engine

app = Flask(__name__)

# リクエストの受付
@app.route("/", methods=["POST", "GET"])
def index():
    # Query Engineの取得
    query_engine = get_query_engine()

    if request.method == "POST":
        question=request.form["question"]
        # Formから質問が送信された場合は回答を検索
        response = query_engine.query(question)
        source = response.get_formatted_sources()
    else:
        question = ""
        response = "."
        source = ""

    # テンプレートに受け渡し
    html = render_template("index.html", 
                           response=response, question=question, source=source)
    return html

if __name__ == "__main__":
    app.run(port=5000)
