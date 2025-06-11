import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader

VECTOR_DB_PATH = "faiss_index"

def create_vector_db(): #PDF 파일 읽어와서 chunk로 스플릿하고 vector db 생성
    loader = PyPDFLoader("news_weather.pdf")
    docs = loader.load()
    print(f"문서의 수: {len(docs)}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    print(f"split size: {len(splits)}")

    embeddings = OllamaEmbeddings(model="llama3.1")

    vector_store = FAISS.from_documents(
        documents=splits,
        embedding=embeddings,
    )

    vector_store.save_local(VECTOR_DB_PATH)

    return vector_store

if os.path.exists(VECTOR_DB_PATH):
    print("기존 벡터 DB를 로드합니다.")
    embeddings = OllamaEmbeddings(model="llama3.1")
    vector_store = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True # 믿을 수 있는 소스임을 확인
    )
else:
    print("새로운 벡터 DB를 생성합니다.")
    vector_store = create_vector_db()


retriever = vector_store.as_retriever()

prompt = PromptTemplate.from_template(
"""당신은 질문-답변(Question-Answering)을 수행하는 AI 어시스턴트입니다. 당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다` 라고 답하세요.
질문과 관련성이 높은 내용만 답변하고 추측된 내용을 생성하지 마세요. 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.
#Question: 
{question} 
#Context: 
{context} 
#Answer:"""
)

llm = Ollama(
    model="llama3.1",
    temperature=0
)

chain = prompt | llm | StrOutputParser()

while True:
    question = input("\n\n당신: ")
    if question == "끝" or question == "exit":
        break
    # 쿼리 처리 (Query-Retriever) : 벡터 DB 에서 참고할 문서 검색
    retrieved_docs = retriever.invoke(question)
    print(f"retrieved size: {len(retrieved_docs)}")
    combined_docs = "\n\n".join(doc.page_content for doc in retrieved_docs)
    # 검색된 문서를 첨부해서 PROMPT 생성
    formatted_prompt = {"context": combined_docs, "question": question}
    # 체인을 실행하고 결과를 stream 형태로 출력
    result = ""
    for chunk in chain.stream(formatted_prompt):
        print(chunk, end="", flush=True)
        result += chunk