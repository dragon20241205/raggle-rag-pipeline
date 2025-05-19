import json
import sys
import time

from dotenv import load_dotenv
import requests
import pdfplumber

from langchain import callbacks
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
import pymupdf4llm
from pypdf import PdfReader
import numpy as np
from io import BytesIO
from PIL import Image
import fitz  # PyMuPDF
import easyocr


# ==============================================================================
# !!! 警告 !!!: 以下の変数を変更しないでください。
# ==============================================================================
model = "gpt-4o-mini"

# コンペではURLでしたが、念のため置換
pdf_file_urls = [
    "pdf_1.pdf",
    "pdf_2.pdf",
    "pdf_3.pdf",
    "pdf_4.pdf",
    "pdf_5.pdf",
]
# ==============================================================================


# ==============================================================================
# この関数を編集して、あなたの RAG パイプラインを実装してください。
# !!! 注意 !!!: デバッグ過程は標準出力に出力しないでください。
# ==============================================================================
def rag_implementation(question: str) -> str:
    """
    ロート製薬の製品・企業情報に関する質問に対して回答を生成する関数
    この関数は与えられた質問に対してRAGパイプラインを用いて回答を生成します。

    Args:
        question (str): ロート製薬の製品・企業情報に関する質問文字列

    Returns:
        answer (str): 質問に対する回答

    Note:
        - デバッグ出力は標準出力に出力しないでください
        - model 変数 と pdf_file_urls 変数は編集しないでください
        - 回答は日本語で生成してください
    """
    def download_and_load_pdfs(urls: list) -> list:
        """
        PDFファイルをダウンロードし、ファイルの種類に応じた前処理を行い、テキストを抽出する関数。

        Args:
            urls (list): PDFファイルのURLリスト

        Returns:
            documents (list): PDFファイルのテキストデータを含むDocumentオブジェクトのリスト

        Raises:
            Exception: ダウンロードまたは読み込みに失敗した場合に発生する例外

        Examples:
            >>> urls = ["https://example.com/example.pdf"]
            >>> download_and_load_pdfs(urls)
            [Document(page_content="...", metadata={"source": "https://example.com/example.pdf"})]
        """
        try:
            def download_pdf(url, save_path):
                response = requests.get(url)
                if response.status_code == 200:
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                else:
                    raise Exception(f"Failed to download {url}")
            
            documents = []
            for i, url in enumerate(urls):
                tmp_path = f"pdf_{i}.pdf"
                download_pdf(url, tmp_path)

                # ファイルごとの処理
                if "Financial_Statements_2023" in url or "Shibata_et_al_Research_Article" in url:
                    # Financial_Statements_2023.pdfまたはShibata_et_al_Research_Article.pdfの処理（pymupdf4llmを使用）
                    md_text = pymupdf4llm.to_markdown(tmp_path)
                    documents.append(Document(page_content=md_text, metadata={"source": url}))


                elif "Hada_Labo_Gokujun_Lotion_Overview" in url:
                    # Hada_Labo_Gokujun_Lotion_Overview.pdf の処理

                    reader = PdfReader(tmp_path)
                    texts = []
                    for page in reader.pages:
                        text = page.extract_text(extraction_mode="layout", layout_mode_space_vertically=False)
                        if text:
                            texts.append(text)

                    chat_model = ChatOpenAI(model=model, temperature=0)
                    formatting_prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", "PDFから抽出された生のテキストを段落構成に整形してください。"),
                            ("human", "{text}")
                        ]
                    )
                    formatting_chain = formatting_prompt|chat_model|StrOutputParser()

                    for page_text in texts:
                        result = formatting_chain.invoke({"text": page_text})
                        documents.append(Document(page_content=result, metadata={"source": url}))



                elif "V_Rohto_Premium_Product_Information" in url: 
                    # V_Rohto_Premium_Product_Information.pdf の処理
                    reader = easyocr.Reader(['ja', 'en'])  # 日本語と英語対応
                    text = ""

                    # PyMuPDF で PDF を画像化
                    pdf_document = fitz.open(tmp_path)
                    for page_num in range(len(pdf_document)):
                        page = pdf_document[page_num]
                        pix = page.get_pixmap(dpi=300)  # 高解像度で画像化
                        img_data = BytesIO(pix.tobytes("png"))  # メモリ上に保存

                        # PIL 画像を開き、NumPy 配列に変換
                        image = Image.open(img_data)
                        image_np = np.array(image)

                        # OCR を実行
                        ocr_results = reader.readtext(image_np, batch_size=8)
                        page_text = " ".join([result[1] for result in ocr_results])  # 抽出テキストを結合
                        text += page_text + "\n"

                    pdf_document.close()
                    documents.append(Document(page_content=text, metadata={"source": url}))

                elif "Well-Being_Report_2024" in url:
                    # Well-Being_Report_2024.pdf の処理
                    with pdfplumber.open(tmp_path) as pdf:
                        full_text = ""
                        for page in pdf.pages:
                            text = page.extract_text()
                            if text:
                                full_text += text + "\n"
                        documents.append(Document(page_content=full_text, metadata={"source": url}))

                else:
                    # サポートされていないファイルタイプの処理
                    raise Exception(f"Unsupported file type or missing handling logic for {url}")

            return documents

        except Exception as e:
            raise Exception(f"Error processing PDFs: {e}")

   
    def create_chroma_vectorstore(docs: list) -> Chroma:
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splitted_docs = []
            for doc in docs:
                chunks = text_splitter.split_text(doc.page_content)
                for chunk in chunks:
                    splitted_docs.append(Document(page_content=chunk, metadata=doc.metadata))

            embedding_function = OpenAIEmbeddings()

            vectorstore = Chroma.from_documents(
                splitted_docs,
                embedding_function,
            )
            return vectorstore
        except Exception as e:
            raise Exception(f"Error creating Chroma vectorstore: {e}")
        
    def create_bm25_retriever(docs: list) -> BM25Retriever:
        def preprocess_func(text):
            def generate_character_ngrams(text, i, j):
                ngrams = []
                for n in range(i, j + 1):
                    for k in range(len(text) - n + 1):
                        ngrams.append(text[k:k + n])
                return list(set(ngrams))
            return generate_character_ngrams(text, 1, 3)

        return BM25Retriever.from_documents(
            docs,
            preprocess_func=preprocess_func,
        )

    def reciprocal_rank_fusion(
        retriever_outputs: list[list[Document]],
        k: int = 60,
        max_context_length: int = 2000  # 最大トークン長
    ) -> str:
        content_score_mapping = {}

        for docs in retriever_outputs:
            for rank, doc in enumerate(docs):
                content = doc.page_content
                if content not in content_score_mapping:
                    content_score_mapping[content] = 0
                content_score_mapping[content] += 1 / (rank + k)
        ranked = sorted(content_score_mapping.items(), key=lambda x: x[1], reverse=True)

        # 結果を結合して1つの文字列にする
        combined_content = "\n".join([content for content, _ in ranked])

        # 最大トークン長を超えないように制限
        if len(combined_content) > max_context_length:
            return combined_content[:max_context_length] + "..."
        return combined_content


    docs = download_and_load_pdfs(pdf_file_urls)
    chroma_db = create_chroma_vectorstore(docs)
    chroma_retriever = chroma_db.as_retriever(search_kwargs={"k": 10})

    bm25_retriever = create_bm25_retriever(docs).with_config({"k": 10})

    hybrid_retriever = (
        RunnableParallel({
            "chroma_documents": chroma_retriever,
            "bm25_documents": bm25_retriever,
        })
        | (lambda x: [x["chroma_documents"], x["bm25_documents"]])
        | reciprocal_rank_fusion
    )

    prompt = ChatPromptTemplate.from_template('''\
    以下の参考文章を踏まえて、質問に簡潔に回答してください。回答には必要に応じて数値や単位を明記してください。
                                            
    参考文章:"""
    {context}
    """
                                              
    質問:
    {question}
    ''')

    chat = ChatOpenAI(model=model,temperature=0)

    hybrid_rag_chain = (
        {
            "question": RunnablePassthrough(),
            "context": hybrid_retriever,
        }
        | prompt | chat | StrOutputParser()
    )

    answer = hybrid_rag_chain.invoke(question)

    return answer


# ==============================================================================


# ==============================================================================
# !!! 警告 !!!: 以下の関数を編集しないでください。
# ==============================================================================
def main(question: str):
    with callbacks.collect_runs() as cb:
        result = rag_implementation(question)
        for attempt in range(2):  # 最大2回試行
            try:
                run_id = cb.traced_runs[0].id
                break
            except IndexError:
                if attempt == 0:  # 1回目の失敗時のみ
                    time.sleep(3)  # 3秒待機して再試行
                else:  # 2回目も失敗した場合
                    raise RuntimeError("Failed to get run_id after 2 attempts")

    output = {"result": result, "run_id": str(run_id)}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    load_dotenv()

    if len(sys.argv) > 1:
        question = sys.argv[1]
        main(question)
    else:
        print("Please provide a question as a command-line argument.")
        sys.exit(1)
# ==============================================================================
