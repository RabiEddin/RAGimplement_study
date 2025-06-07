import bs4 # 웹 크롤링

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# 문서 로딩
loader = WebBaseLoader(
    web_paths=("https://www.bbc.com/korean/articles/cl4yml4l6j1o",), # 문서 경로
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "div", # div 태그 중에 아래 2개의 클래스 중 하나라도 가지고 있으면 파싱하기
            attrs={"class": ["bbc-1cvxiy9", "bbc-fa0wmp"]},
        )
    ),
)
docs = loader.load()
print(f"문서의 수: {len(docs)}")

# 문서 분할
# 청크 단위로 자르는 이유: 모델 입력 길이의 제한, 청크를 오버랩 하는 이유: 문맥의 일관성을 유지
# 청크로 나누는 두번째 이유: 내용 길이가 길면 중간 내용을 잊어버림(lost in the middle)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)
print(f"split size: {len(splits)}")

# 임베딩 생성 - 임베딩 = 텍스트 -> 벡터 변환
embeddings = OllamaEmbeddings(model="llama3.1")

# 벡터 저장소 구축
vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)

# 쿼리 저장소 검색을 위한 retriever 생성
retriever = vector_store.as_retriever()

# PROMPT Template 생성
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

#Ollama 초기화, temperature는 0에 가까울수록 같은 입력에도 항상 같은 답만 반환, 값이 높을수록 다양한 답을 반환
llm = Ollama(
    model="llama3.1",
    temperature=0,
    base_url="http://localhost:11434"
)

# Langchain의 파이프라인 구성 문법을 활용
# 앞에서부터 순서대로 연결
# StrOutputParser는 LangChain의 출력 파서 중 하나
chain = prompt | llm | StrOutputParser()

# 테스트할 질문
question = "극한 호우의 원인은 무엇인가?"

# Query-Retriever: 벡터 DB에서 참고할 문서 검색
# question을 벡터화하고 FAISS에서 유사도 검색해서 반환, FAISS는 내적 계산으로 유사도 검사 + Top-k는 4가 기본 설정
retrieved_docs = retriever.invoke(question)
print(f"retrieved size: {len(retrieved_docs)}")
# page_content는 retriever로 가져온 doc를 청크단위로 각각임. DOC A -> doc1.page_content, doc2.page_content, ...
combined_docs = "\n\n".join(doc.page_content for doc in retrieved_docs)

# 6. 검색된 문서를 첨부해서 PROMPT 생성
formatted_prompt = {"context": combined_docs, "question": question}
# 7. LLM에 질문
for chunk in chain.stream(formatted_prompt):
    print(chunk, end="", flush=True)