import json
import sys
import urllib.error
import urllib.request

import chromadb
from chromadb.utils import embedding_functions

import golden
import metrics
import search


OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "bge-m3"

COLLECTION_NAME = "phase4_eval"


def check_ollama():
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=5) as res:
            data = json.loads(res.read())
    except (urllib.error.URLError, OSError):
        print("[준비 실패] Ollama 서버에 연결하지 못했습니다.")
        print("  서버 실행:  brew services start ollama")
        return False

    installed = []
    for model in data.get("models", []):
        installed.append(model.get("name", ""))

    has_model = False
    for name in installed:
        if name.startswith(EMBED_MODEL):
            has_model = True

    if not has_model:
        print("[준비 실패] 임베딩 모델 '%s' 가 없습니다." % EMBED_MODEL)
        print("  모델 준비:  ollama pull %s" % EMBED_MODEL)
        return False

    print("[준비 완료] Ollama 서버 연결 OK, 임베딩 모델 '%s' 확인." % EMBED_MODEL)
    return True


def reset_collection(client, ollama_ef):
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


def load_corpus(collection):
    ids = []
    documents = []
    for doc_id, text in golden.CORPUS:
        ids.append(doc_id)
        documents.append(text)
    collection.add(ids=ids, documents=documents)


def first_correct_rank(retrieved_ids, relevant_ids):
    for i in range(len(retrieved_ids)):
        if retrieved_ids[i] in relevant_ids:
            return i + 1
    return None


def evaluate(col, search_fn):
    recall1_sum = 0.0
    recall3_sum = 0.0
    mrr_sum = 0.0
    ndcg3_sum = 0.0

    per_query = []

    for question, relevant_ids in golden.GOLDEN:
        retrieved = search_fn(col, question, k=5)

        r1 = metrics.recall_at_k(retrieved, relevant_ids, 1)
        r3 = metrics.recall_at_k(retrieved, relevant_ids, 3)
        rr = metrics.mrr(retrieved, relevant_ids)
        nd = metrics.ndcg_at_k(retrieved, relevant_ids, 3)

        recall1_sum = recall1_sum + r1
        recall3_sum = recall3_sum + r3
        mrr_sum = mrr_sum + rr
        ndcg3_sum = ndcg3_sum + nd

        rank = first_correct_rank(retrieved, relevant_ids)
        per_query.append({"question": question, "relevant": relevant_ids, "rank": rank})

    n = len(golden.GOLDEN)
    averages = {
        "recall@1": recall1_sum / n,
        "recall@3": recall3_sum / n,
        "MRR": mrr_sum / n,
        "nDCG@3": ndcg3_sum / n,
    }
    return averages, per_query


def print_rank_line(item):
    if item["rank"] is None:
        rank_text = "정답 없음(5위 밖)"
    else:
        rank_text = "%d위" % item["rank"]
    print("    - %s -> 정답 %s -> %s" % (item["question"], item["relevant"], rank_text))


def print_metrics(averages):
    print("    recall@1 = %.3f" % averages["recall@1"])
    print("    recall@3 = %.3f" % averages["recall@3"])
    print("    MRR      = %.3f" % averages["MRR"])
    print("    nDCG@3   = %.3f" % averages["nDCG@3"])


def step1_prepare_and_load(client, ollama_ef):
    print()
    print("[1] 준비 & 코퍼스 적재")

    collection = reset_collection(client, ollama_ef)
    load_corpus(collection)

    print("  검색 대상 문서를 컬렉션에 담았습니다.")
    print("  총 문서 수: %d 개" % collection.count())
    return collection


def step2_golden_intro():
    print()
    print("[2] 골든셋 소개")
    print("  골든셋 = (질문, 정답 문서 id) 짝을 사람이 미리 정해 둔 정답표입니다.")
    print("  이 정답표가 있어야 검색 결과가 맞았는지 '점수'로 잴 수 있습니다.")
    print("  질문 %d개:" % len(golden.GOLDEN))
    for question, relevant_ids in golden.GOLDEN:
        print("    - %s -> %s" % (question, relevant_ids))


def step3_eval_vector(collection):
    print()
    print("[3] 평가: 순수 벡터")
    averages, per_query = evaluate(collection, search.vector_search)
    print("  질문별 첫 정답 순위:")
    for item in per_query:
        print_rank_line(item)
    print("  지표:")
    print_metrics(averages)
    return averages, per_query


def step4_eval_hybrid(collection):
    print()
    print("[4] 평가: 하이브리드")
    averages, per_query = evaluate(collection, search.hybrid_search)
    print("  질문별 첫 정답 순위:")
    for item in per_query:
        print_rank_line(item)
    print("  지표:")
    print_metrics(averages)
    return averages, per_query


def step5_compare(vector_avg, vector_pq, hybrid_avg, hybrid_pq):
    print()
    print("[5] 튜닝 전후 비교 (순수 벡터 -> 하이브리드)")

    print("  지표            순수 벡터   하이브리드")
    metric_keys = ["recall@1", "recall@3", "MRR", "nDCG@3"]
    for key in metric_keys:
        print("    %-10s   %.3f       %.3f" % (key, vector_avg[key], hybrid_avg[key]))

    print()
    print("  어떤 질문에서 갈렸는가:")
    for i in range(len(golden.GOLDEN)):
        v_rank = vector_pq[i]["rank"]
        h_rank = hybrid_pq[i]["rank"]
        if v_rank != h_rank:
            question = golden.GOLDEN[i][0]
            print("    - %s : 벡터 %s위 -> 하이브리드 %s위" % (question, v_rank, h_rank))
    print("    (E-402는 순수 벡터에서 4위로 밀렸다가 하이브리드에서 1위로 올라옵니다.)")

    print()
    print("  정직한 해석:")
    print("    bge-m3(좋은 다국어 임베딩)는 이미 대부분 정답을 1위로 잘 찾습니다.")
    print("    하이브리드는 벡터가 놓친 exact-token 케이스(E-402) 하나를 살려")
    print("    recall@1을 0.83 -> 1.0으로 올릴 뿐입니다.")
    print("    즉 '하이브리드가 항상 낫다'는 착각일 수 있고,")
    print("    개선인지 아닌지는 반드시 '측정'해야 압니다 — 그게 평가의 존재 이유입니다.")


def main():
    if not check_ollama():
        sys.exit(1)

    client = chromadb.PersistentClient(path="./chroma_db")
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url=OLLAMA_URL,
        model_name=EMBED_MODEL,
    )

    collection = step1_prepare_and_load(client, ollama_ef)
    step2_golden_intro()
    vector_avg, vector_pq = step3_eval_vector(collection)
    hybrid_avg, hybrid_pq = step4_eval_hybrid(collection)
    step5_compare(vector_avg, vector_pq, hybrid_avg, hybrid_pq)


if __name__ == "__main__":
    main()
