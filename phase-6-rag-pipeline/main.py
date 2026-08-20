import json
import sys
import urllib.error
import urllib.request

import chromadb
from chromadb.utils import embedding_functions

import documents
import pipeline


OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "bge-m3"
CHAT_MODEL = "qwen2.5:7b"
COLLECTION_NAME = "phase6_support"
MAX_SIZE = 90
TOP_K = 3


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


def main():
    if not check_ollama():
        sys.exit(1)

    client = chromadb.PersistentClient(path="./chroma_db")
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url=OLLAMA_URL,
        model_name=EMBED_MODEL,
    )
    collection = build_collection(client, ollama_ef)

    print()
    print("[1] 인제스트: 문서를 청킹해 벡터 DB에 저장")
    chunks = pipeline.ingest(collection, documents.DOC, MAX_SIZE)
    print("  recursive_split(max=%d) 결과 청크 %d개:" % (MAX_SIZE, len(chunks)))
    for i in range(len(chunks)):
        print("    [c%d] %s" % (i, chunks[i]))

    print()
    print("[2] RAG 질의응답: 검색(R) -> 증강(A) -> 생성(G)")
    for question in documents.QUESTIONS:
        result = pipeline.answer(collection, question, TOP_K)
        print()
        print("  질문: %s" % question)
        print("  검색된 문맥(sources):")
        for source_id, source_doc in result["sources"]:
            print("    [%s] %s" % (source_id, source_doc[:30]))
        print("  LLM 답변: %s" % result["answer"])

    print()
    print("[3] 정리")
    print("  이게 RAG 전체 루프다: 검색(R)으로 관련 청크를 찾고,")
    print("  그 청크를 프롬프트 [문맥]에 넣어 증강(A)하고, LLM이 생성(G)한다.")
    print("  문맥에 근거가 없으면(주차장 질문) LLM은 지어내지 않고")
    print("  '문서에서 찾을 수 없습니다.'로 거부한다.")
    print("  근거가 문맥에 있을 때만 답하게 하는 grounding이 할루시네이션을 줄인다.")


if __name__ == "__main__":
    main()