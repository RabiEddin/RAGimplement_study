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

print(chain2.invoke({"korean_world": "한옥"}))