#Few-shot: 언어 모델에 몇 가지 예시를 제공하여 특정 작업을 수행하도록 유도하는 기법

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_ollama import OllamaEmbeddings

lllm = Ollama(
    model="llama3.1",
    temperature=0
)

example_prompt = PromptTemplate.from_template("Question: {question}\nAnswer: {answer}")

examples = [
    {
        "question": "지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?",
        "answer": "지구 대기의 약 78%를 차지하는 질소입니다."
    },
    {
        "question": "광합성에 필요한 주요 요소들은 무엇인가요?",
        "answer": "광합성에 필요한 주요 요소는 빛, 이산화탄소, 물입니다."
    },
    {
        "question": "피타고라스 정리를 설명해주세요.",
        "answer": "피타고라스 정리는 직각삼각형에서 빗변의 제곱이 다른 두 변의 제곱의 합과 같다는 것입니다."
    },
    {
        "question": "지구의 자전 주기는 얼마인가요?",
        "answer": "지구의 자전 주기는 약 24시간(정확히는 23시간 56분 4초)입니다."
    },
    {
        "question": "DNA의 기본 구조를 간단히 설명해주세요.",
        "answer": "DNA는 두 개의 폴리뉴클레오티드 사슬이 이중 나선 구조를 이루고 있습니다."
    },
    {
        "question": "원주율(π)의 정의는 무엇인가요?",
        "answer": "원주율(π)은 원의 지름에 대한 원의 둘레의 비율입니다."
    }
]

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}\nAnswer: ", #사용자로부터 입력받은 질문을 삽입할 부분으로 보통 LLM이 실제로 추론하거나 답을 생성할 부분.
    input_variables=["input"],
)

# 이렇게 하면 prompt에 있는 예시들이랑 suffix들이 출력됨.
# print(prompt.invoke({"input": "화성의 표면이 붉은 이유는 무엇인가요?"}).to_string())

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,                           # 사용할 예제들
    OllamaEmbeddings(model="llama3.1"), # 임베딩 모델
    Chroma,                             # 벡터 저장소
    k=1,                                # 선택할 예제 수
)

# 새로운 질문에 대해 가장 유사한 예제를 선택
question = "화성의 표면이 붉은 이유는 무엇인가요?"
selected_examples = example_selector.select_examples({"question": question}) # 선택된 유사한 예제
print(f"입력과 가장 유사한 예제: {question}")
for example in selected_examples:
    print("\n")
    for k, v in example.items():
        print(f"{k}: {v}")

