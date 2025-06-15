# Few-shot에서 참고할 예제들 고정된 예제로 사용하기

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser

# 참고할 예제들 정의
examples = [
    {"input": "지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?", "output": "질소입니다."},
    {"input": "광합성에 필요한 주요 요소들은 무엇인가요?", "output": "빛, 이산화탄소, 물입니다."},
]

# 예제 프롬프트 템플릿 정의
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

# Few-shot 프롬프트 템플릿 생성
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 최종 프롬프트 템플릿 생성
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 과학과 수학에 대해 잘 아는 교육자입니다."), # -> ai한테 역할을 설정해주는 거
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

llm = Ollama (
    model="llama3.1",
    temperature = 0,
)

chain = final_prompt | llm | StrOutputParser()

result = chain.invoke({"input": "지구의 자전 주기는 얼마인가요?"})
print(result)
