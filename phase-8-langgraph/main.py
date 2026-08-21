import json
import sys
import urllib.error
import urllib.request

import chromadb
from chromadb.utils import embedding_functions

import documents
import agent


OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "bge-m3"
CHAT_MODEL = "qwen2.5:7b"
COLLECTION_NAME = "phase8_agentic"
MAX_SIZE = 90


def check_ollama():
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=5) as res:
            data = json.loads(res.read())
    except (urllib.error.URLError, OSError):
        print("[준비 실패] Ollama 서버에 연결하지 못했습니다.")
        print("  서버 실행:  brew services start ollama")
        return False

    names = []
    for model in data.get("models", []):
        names.append(model.get("name", ""))

    has_embed = False
    has_chat = False
    for name in names:
        if name.startswith(EMBED_MODEL):
            has_embed = True
        if name.startswith(CHAT_MODEL):
            has_chat = True

    ok = True
    if not has_embed:
        print("[준비 실패] 임베딩 모델 '%s' 가 없습니다." % EMBED_MODEL)
        print("  모델 준비:  ollama pull %s" % EMBED_MODEL)
        ok = False
    if not has_chat:
        print("[준비 실패] 채팅 LLM '%s' 가 없습니다." % CHAT_MODEL)
        print("  모델 준비:  ollama pull %s" % CHAT_MODEL)
        ok = False

    if ok:
        print("[준비 완료] Ollama 연결 OK, 임베딩 '%s' + 채팅 '%s' 확인." % (EMBED_MODEL, CHAT_MODEL))
    return ok


def build_collection(client, ollama_ef):
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def ingest(collection, doc, max_size):
    chunks = agent.recursive_split(doc, max_size)
    ids = []
    for i in range(len(chunks)):
        ids.append("c%d" % i)
    collection.add(ids=ids, documents=chunks)
    return chunks


def main():
    if not check_ollama():
        sys.exit(1)

    client = chromadb.PersistentClient(path="./chroma_db")
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url=OLLAMA_URL,
        model_name=EMBED_MODEL,
    )
    collection = build_collection(client, ollama_ef)
    chunks = ingest(collection, documents.DOC, MAX_SIZE)

    graph = agent.build_graph(collection)

    print()
    print("[1] 그래프 구조")
    print("  노드 4개:")
    print("    retrieve        : 벡터 DB에서 질문에 가까운 청크를 검색한다.")
    print("    grade           : 검색된 청크가 질문에 답할 근거를 담았는지 LLM으로 yes/no 판정한다.")
    print("    transform_query : 근거가 없으면 질문을 검색 잘 되게 다시 쓴다.")
    print("    generate        : 문맥만 근거로 답하고, 근거 없으면 못 찾겠다고 답한다.")
    print("  조건부 라우팅(grade 다음):")
    print("    관련성 yes            -> generate (바로 답)")
    print("    관련성 no             -> transform_query (질문 고쳐 재검색)")
    print("    재시도 %d회 초과       -> generate (포기하고 못 찾겠다고 답)" % agent.MAX_TRIES)
    print("  적재된 청크 %d개 (recursive_split max=%d)" % (len(chunks), MAX_SIZE))

    print()
    print("[2] 실행")
    for question in documents.QUESTIONS:
        print()
        print("=" * 60)
        print("질문: %s" % question)
        print("-" * 60)
        initial = {
            "question": question,
            "documents": [],
            "relevant": "",
            "generation": "",
            "tries": 0,
        }
        out = graph.invoke(initial)
        print("-" * 60)
        print("최종 답변: %s" % out["generation"])
        print("검색 시도(tries): %d" % out["tries"])

    print()
    print("[3] 정리")
    print("  이게 Agentic RAG다. Phase 0~7에서 만든 검색(R)과 생성(G)이 LangGraph '노드'가 되고,")
    print("  에이전트가 검색 결과를 스스로 평가(grade)해서 재검색할지(transform_query)")
    print("  답할지(generate)를 조건부 엣지로 정한다.")
    print("  첫 질문은 근거가 바로 있어 retrieve->grade=yes->generate로 곧장 답했고,")
    print("  둘째 질문은 grade=no라 transform_query로 재검색까지 했지만 여전히 근거가 없어")
    print("  재시도 한도에서 포기하고 '문서에서 찾을 수 없습니다.'로 답했다.")
    print("  검색 결과의 품질을 에이전트가 판단해 흐름을 스스로 바꾸는 것,")
    print("  이게 LangGraph 에이전트 개발의 핵심이고 이 커리큘럼의 종착점이다.")


if __name__ == "__main__":
    main()
