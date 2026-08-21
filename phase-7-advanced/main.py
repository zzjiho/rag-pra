import json
import sys
import urllib.error
import urllib.request

import chromadb
from chromadb.utils import embedding_functions

import advanced
import corpus


OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "bge-m3"
CHAT_MODEL = "qwen2.5:7b"
COLLECTION_NAME = "phase7_advanced"
N_VARIANTS = 4
TOP_K = 3
RERANK_K = 5


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

    ids = []
    docs = []
    for doc_id, text in corpus.CORPUS:
        ids.append(doc_id)
        docs.append(text)
    collection.add(ids=ids, documents=docs)
    return collection


def rank_of(ranked_ids, answer_id):
    for i in range(len(ranked_ids)):
        if ranked_ids[i] == answer_id:
            return i + 1
    return 0


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
    print("[1] Multi-Query Retrieval: 한 질문을 여러 표현으로 물어 검색 누락을 줄인다")
    mq_question = "환불하고 싶은데 어떻게 해요"
    print("  질문: %s" % mq_question)
    variants = advanced.multi_query(mq_question, N_VARIANTS)
    print("  생성된 변형 질문:")
    for i in range(len(variants)):
        print("    - %s" % variants[i])
    merged_ids = advanced.multi_query_search(collection, mq_question, N_VARIANTS, TOP_K)
    print("  변형 질문들의 검색 결과를 합친 후보 id (중복 제거): %s" % merged_ids)
    print("  하나의 표현으로만 검색하면 놓칠 문서를, 여러 표현으로 물어 후보로 끌어온다.")

    print()
    print("[2] HyDE: 질문 대신 '가상 답변'을 임베딩해 문서와 어휘를 맞춘다")
    hyde_question = "노트북 배터리 얼마나 가요"
    print("  질문: %s" % hyde_question)
    hypo_text = advanced.hyde(hyde_question)
    print("  LLM이 상상한 가상 답변:")
    print("    %s" % hypo_text)
    hyde_ids, hyde_docs = advanced.hyde_search(collection, hypo_text, TOP_K)
    print("  가상 답변으로 검색한 결과 id: %s" % hyde_ids)
    print("  질문은 짧고 어휘가 부족하지만, 가상 답변은 실제 문서와 어휘가 비슷해져 검색이 잘 된다.")

    print()
    print("[3] 재랭킹 (2단계, 측정): 넓게 뽑고(벡터) -> 정밀 정렬(LLM)")
    be_recall_sum = 0.0
    be_mrr_sum = 0.0
    rr_recall_sum = 0.0
    rr_mrr_sum = 0.0
    for question, answer_id in corpus.GOLDEN:
        ids, docs = advanced.embed_search(collection, question, RERANK_K)
        reranked = advanced.llm_rerank(question, ids, docs)

        be_recall_sum = be_recall_sum + advanced.recall_at_1(ids, answer_id)
        be_mrr_sum = be_mrr_sum + advanced.mrr(ids, answer_id)
        rr_recall_sum = rr_recall_sum + advanced.recall_at_1(reranked, answer_id)
        rr_mrr_sum = rr_mrr_sum + advanced.mrr(reranked, answer_id)

        print()
        print("  질문: %s   (정답: %s)" % (question, answer_id))
        print("    (a) 벡터 검색 top-5: %s" % ids)
        print("        정답 순위: %d위" % rank_of(ids, answer_id))
        print("    (b) LLM 재랭킹 후 : %s" % reranked)
        print("        정답 순위: %d위" % rank_of(reranked, answer_id))

    n = len(corpus.GOLDEN)
    print()
    print("  === 지표 비교 (recall@1 / MRR) ===")
    print("    벡터 검색만  : recall@1=%.3f  MRR=%.3f" % (be_recall_sum / n, be_mrr_sum / n))
    print("    LLM 재랭킹 후: recall@1=%.3f  MRR=%.3f" % (rr_recall_sum / n, rr_mrr_sum / n))
    print("  벡터 검색은 빠르고 싸지만 정답을 상위로 못 올리는 경우가 있다(1단계로 후보만 넓게 확보).")
    print("  느리지만 정확한 재랭킹이 질문-문서 관련도를 다시 따져 top을 정밀 정렬한다(2단계).")

    print()
    print("[4] 정리")
    print("  Multi-Query·HyDE·재랭킹은 검색 recall과 정확도를 끌어올리는 고급 패턴이다.")
    print("  실무 재랭킹은 전용 cross-encoder(예: bge-reranker-v2-m3)를 쓴다.")
    print("  설치가 무거워 여기서는 이미 있는 LLM으로 대체했다(LLM-as-reranker).")
    print("  Self-Query(자연어->메타 필터)와 Conversational RAG(대화 이력 기반 질문 재작성)는")
    print("  코드 데모 대신 README에서 개념으로 다룬다.")


if __name__ == "__main__":
    main()