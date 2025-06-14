from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

llm = Ollama(
    model="llama3.1",
    temperature=0
)

prompt = PromptTemplate.from_template("""You are an expert in motersports. Answer the question. 
#<Question>: 
{question}
#Answer:"""
"")

chain = prompt | llm | StrOutputParser()

print(chain.invoke({"question": "who is the F1 winner of the 2022 driver championship?"}))