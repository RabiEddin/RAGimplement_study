from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = Ollama (
    model="llama3.1",
    temperature = 0
)

# ChatpromptTemplate는 대화형 상황에서 여러 메시지 입력을 기반으로 단일 메시지 응답을 생성하는데 사용.
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "This system can be answered about Formula 1."),
    ("user", "{user_input}"),
])

chain = chat_prompt | llm | StrOutputParser()

for chunk in chain.stream({"user_input": "max_verstappen explanation in korean"}):
    print(chunk, end="", flush=True)



