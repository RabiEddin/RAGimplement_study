from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_ollama import OllamaEmbeddings

examples = [
    {"input": "지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?", "output": "질소입니다."},
    {"input": "광합성에 필요한 주요 요소들은 무엇인가요?", "output": "빛, 이산화탄소, 물입니다."},
    {"input": "피타고라스 정리를 설명해주세요.", "output": "직각삼각형에서 빗변의 제곱은 다른 두 변의 제곱의 합과 같습니다."},
    {"input": "DNA의 기본 구조를 간단히 설명해주세요.", "output": "DNA는 이중 나선 구조를 가진 핵산입니다."},
    {"input": "원주율(π)의 정의는 무엇인가요?", "output": "원의 둘레와 지름의 비율입니다."},
]

vector_db = ["".join(example.values()) for example in examples]
embeddings = OllamaEmbeddings()
vectorstore = Chroma.from_texts(vector_db, embeddings, metadatas=examples)

