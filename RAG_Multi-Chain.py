from langchain_community.llms import Ollama
from langchain_core.prompts import  PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = Ollama(
    model="llama3.1",
    temperature=0
)

prompt1 = PromptTemplate.from_template("translates {korean_world} to English.")
prompt2 = PromptTemplate.from_template("explain {english_world} to me in korean language.")

chain1 = prompt1 | llm | StrOutputParser()

print(chain1.invoke({"korean_world": "한옥"}))

chain2 = {"english_world": chain1} | prompt2 | llm | StrOutputParser()

print(chain2.invoke({"korean_world": "한옥"})) # invoke: llm으로부터 결과 반환받아서 한 번에 출력

for chunk in chain2.stream({"korean_world": "한옥"}): # stream: llm으로부터 결과 반환받아서 실시간으로 출력.
    print(chunk, end="", flush=True)