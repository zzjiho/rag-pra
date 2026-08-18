import json
import sys
import urllib.error
import urllib.request

import chromadb
from chromadb.utils import embedding_functions


OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "bge-m3"

COLLECTION_NAME = "policy_and_errors"

RRF_K = 60


# 정책 FAQ 문단 (메타데이터 필터 데모용). 오류 문서가 아니므로 code 는 "NONE".
FAQ_DOCS = [
    {
        "id": "faq-refund-1",
        "text": "환불을 원하시면 마이페이지 주문 내역에서 환불 신청 버튼을 누르고 사유를 선택해 접수하시면 됩니다. 승인되면 3영업일 이내에 결제하신 수단으로 환불됩니다.",
        "topic": "환불",
    },
    {
        "id": "faq-refund-2",
        "text": "환불은 상품을 받으신 후 7일 이내에 신청하신 건에 한해 가능합니다. 단순 변심의 경우 왕복 배송비는 고객님이 부담하십니다.",
        "topic": "환불",
    },
    {
        "id": "faq-exchange-1",
        "text": "교환 신청은 상품에 하자가 있거나 다른 옵션을 원하실 때 마이페이지에서 교환 신청을 눌러 접수하실 수 있습니다. 회수 확인 후 새 상품이 발송됩니다.",
        "topic": "교환",
    },
    {
        "id": "faq-delivery-1",
        "text": "주문하신 상품은 결제 완료 후 보통 2~3일 이내에 출고되어 배송됩니다. 배송 조회는 마이페이지 주문 내역에서 확인하실 수 있습니다.",
        "topic": "배송",
    },
    {
        "id": "faq-membership-1",
        "text": "멤버십 가입 신청은 앱 설정 메뉴에서 하실 수 있습니다. 가입하시면 무료 배송과 포인트 적립 등 다양한 혜택을 받으실 수 있습니다.",
        "topic": "멤버십",
    },
]

# 오류 코드 문서 (하이브리드 데모용). 문장은 "오류 코드 E-40X: ...설명..." 형태.
ERROR_DOCS = [
    {
        "id": "err-401",
        "text": "오류 코드 E-401: 로그인 인증이 만료되었습니다. 다시 로그인하시면 됩니다.",
        "code": "E-401",
    },
    {
        "id": "err-402",
        "text": "오류 코드 E-402: 결제가 거절되었습니다. 카드 한도와 결제 정보를 확인한 뒤 다시 시도하시면 해결됩니다.",
        "code": "E-402",
    },
    {
        "id": "err-403",
        "text": "오류 코드 E-403: 접근 권한이 없습니다. 계정 권한을 관리자에게 요청하세요.",
        "code": "E-403",
    },
    {
        "id": "err-404",
        "text": "오류 코드 E-404: 요청하신 페이지를 찾을 수 없습니다. 주소가 올바른지 확인하세요.",
        "code": "E-404",
    },
]


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


def step1_prepare_and_load(client, ollama_ef):
    print()
    print("[1] 준비 & 적재")

    collection = reset_collection(client, ollama_ef)

    ids = []
    documents = []
    metadatas = []

    for doc in FAQ_DOCS:
        ids.append(doc["id"])
        documents.append(doc["text"])
        metadatas.append({"topic": doc["topic"], "code": "NONE"})

    for doc in ERROR_DOCS:
        ids.append(doc["id"])
        documents.append(doc["text"])
        metadatas.append({"topic": "오류", "code": doc["code"]})

    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    print("  한 컬렉션에 두 종류의 문서를 함께 담았습니다.")
    print("    - 정책 FAQ 문단 %d개 (topic=환불/교환/배송/멤버십, code=NONE)" % len(FAQ_DOCS))
    print("    - 오류 코드 문서 %d개 (topic=오류, code=E-40X)" % len(ERROR_DOCS))
    print("  총 문서 수: %d 개" % collection.count())

    return collection


def step2_metadata_filter(collection):
    print()
    print("[2] 메타데이터 필터 (where)")

    question = "신청 방법 알려줘"
    print("  질문: %s" % question)

    print()
    print("  (가) 필터 없이 검색 (n_results=3):")
    result = collection.query(query_texts=[question], n_results=3)
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    for rank in range(len(docs)):
        print("    %d위 | topic=%s" % (rank + 1, metas[rank]["topic"]))
        print("         %s" % docs[rank])
    print("  -> 환불/교환/멤버십 등 '신청'이 들어간 여러 주제가 섞여 나옵니다.")

    print()
    print("  (나) where={\"topic\":\"환불\"} 로 검색 (n_results=3):")
    result = collection.query(query_texts=[question], n_results=3, where={"topic": "환불"})
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    for rank in range(len(docs)):
        print("    %d위 | topic=%s" % (rank + 1, metas[rank]["topic"]))
        print("         %s" % docs[rank])
    print("  -> topic=환불 문서만 남습니다. 나머지 주제는 아예 후보에서 빠집니다.")
    print("  정리: where 는 구조화된 필드(metadata)로 검색 공간을 먼저 좁힌 뒤,")
    print("        그 안에서만 의미 검색을 합니다. 주제가 분명할 때 잡음을 크게 줄여 줍니다.")


def step3_keyword_filter(collection):
    print()
    print("[3] 키워드 필터 (where_document $contains)")

    question = "오류 해결 방법"
    print("  질문: %s" % question)

    print()
    print("  (가) where_document={\"$contains\":\"E-402\"} (본문에 'E-402'가 들어간 문서만):")
    result = collection.query(
        query_texts=[question],
        n_results=5,
        where_document={"$contains": "E-402"},
    )
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    for rank in range(len(docs)):
        print("    %d위 | code=%s" % (rank + 1, metas[rank]["code"]))
        print("         %s" % docs[rank])
    print("  -> 본문에 정확히 'E-402'가 있는 문서만 나옵니다.")

    print()
    print("  (나) where={\"code\":\"E-402\"} (metadata 필드 매칭):")
    result = collection.query(
        query_texts=[question],
        n_results=5,
        where={"code": "E-402"},
    )
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    for rank in range(len(docs)):
        print("    %d위 | code=%s" % (rank + 1, metas[rank]["code"]))
        print("         %s" % docs[rank])
    print("  -> 결과는 같은 E-402 문서 하나입니다. 하지만 걸러낸 방식이 다릅니다.")
    print("  정리: where 는 우리가 저장해 둔 metadata 필드(code)를 매칭하고,")
    print("        where_document 는 문서 '본문' 안의 부분문자열을 매칭합니다.")
    print("        metadata가 없어도 본문만으로 정확한 토큰을 걸러낼 수 있는 게 where_document 입니다.")


def step4_why_hybrid(collection):
    print()
    print("[4] 하이브리드가 필요한 이유")

    question = "E-402 문제가 발생했어요"
    print("  질문: %s" % question)
    print("  순수 벡터 검색만으로 (필터 없이) n_results=4:")

    result = collection.query(query_texts=[question], n_results=4)
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    distances = result["distances"][0]

    for rank in range(len(docs)):
        print("    %d위 | 거리 %.4f | code=%s" % (rank + 1, distances[rank], metas[rank]["code"]))
        print("         %s" % docs[rank])

    top_code = metas[0]["code"]
    print("  -> 1위가 정답 E-402가 아니라 %s 입니다." % top_code)
    print("  문제: 임베딩은 'E-401/402/403/404'처럼 비슷하게 생긴 코드들을 의미로 잘 못 가릅니다.")
    print("        코드·이름·SKU 같은 '정확한 토큰'에는 벡터(의미) 검색이 약합니다.")



def step5_hybrid(collection):
    print()
    print("[5] 하이브리드 검색 (벡터 + 키워드 결합) — 검색기 v2")

    question = "E-402 문제가 발생했어요"
    print("  같은 질문: %s" % question)
    print("  hybrid_search 결과 (RRF 결합, 최종점수 높은 순):")

    results = hybrid_search(collection, question, 4)

    for rank in range(len(results)):
        item = results[rank]
        print(
            "    %d위 | 최종점수 %.5f | 벡터순위=%s 키워드순위=%s | code=%s"
            % (
                rank + 1,
                item["final_score"],
                item["vector_rank"],
                item["keyword_rank"],
                item["code"],
            )
        )
        print("         %s" % item["text"])

    top_code = results[0]["code"]
    print("  -> 이제 1위가 정답 %s 입니다. [4]에서 밀렸던 문서가 위로 올라왔습니다." % top_code)
    print("  교훈: 벡터(의미)는 동의어·문맥에 강하지만 정확한 토큰에 약하고,")
    print("        키워드(정확 일치)는 그 반대입니다.")
    print("        두 순위를 RRF로 합치면 서로의 약점을 메워 줍니다.")



def hybrid_search(collection, query, k):
    # (a) 벡터 순위: 벡터 검색 상위 k개에 1위부터 순위를 매긴다.
    vector_result = collection.query(query_texts=[query], n_results=k)
    vector_ids = vector_result["ids"][0]

    vector_rank = {}
    for i in range(len(vector_ids)):
        doc_id = vector_ids[i]
        vector_rank[doc_id] = i + 1

    # (b) 키워드 순위: 전체 문서를 가져와, query 단어가 본문에 부분문자열로 몇 개 들어있는지 센다.
    all_docs = collection.get()
    all_ids = all_docs["ids"]
    all_texts = all_docs["documents"]

    words = query.split()

    keyword_score = {}
    for i in range(len(all_ids)):
        doc_id = all_ids[i]
        text = all_texts[i]
        score = 0
        for word in words:
            if word in text:
                score = score + 1
        keyword_score[doc_id] = score

    # 0점(단어가 하나도 안 겹치는 문서)은 키워드 후보에서 제외.
    keyword_candidates = []
    for i in range(len(all_ids)):
        doc_id = all_ids[i]
        if keyword_score[doc_id] > 0:
            keyword_candidates.append(doc_id)

    # 점수 높은 순으로 정렬해 1위부터 키워드 순위를 매긴다.
    keyword_candidates.sort(key=lambda doc_id: keyword_score[doc_id], reverse=True)

    keyword_rank = {}
    for i in range(len(keyword_candidates)):
        doc_id = keyword_candidates[i]
        keyword_rank[doc_id] = i + 1

    # (c) RRF: 최종점수 = 1/(RRF_K+벡터순위) + 1/(RRF_K+키워드순위). 한쪽에만 있으면 그 항만 더한다.
    fused_ids = []
    for doc_id in vector_rank:
        fused_ids.append(doc_id)
    for doc_id in keyword_rank:
        if doc_id not in fused_ids:
            fused_ids.append(doc_id)

    final_score = {}
    for doc_id in fused_ids:
        score = 0.0
        if doc_id in vector_rank:
            score = score + 1.0 / (RRF_K + vector_rank[doc_id])
        if doc_id in keyword_rank:
            score = score + 1.0 / (RRF_K + keyword_rank[doc_id])
        final_score[doc_id] = score

    ranked_ids = list(final_score.keys())
    ranked_ids.sort(key=lambda doc_id: final_score[doc_id], reverse=True)

    top_ids = ranked_ids[:k]

    # 출력에 쓰려고 id -> 본문/코드 를 미리 정리해 둔다.
    all_metadatas = all_docs["metadatas"]
    id_to_text = {}
    id_to_code = {}
    for i in range(len(all_ids)):
        id_to_text[all_ids[i]] = all_texts[i]
        id_to_code[all_ids[i]] = all_metadatas[i]["code"]

    results = []
    for doc_id in top_ids:
        item = {
            "id": doc_id,
            "code": id_to_code[doc_id],
            "text": id_to_text[doc_id],
            "final_score": final_score[doc_id],
            "vector_rank": vector_rank.get(doc_id),
            "keyword_rank": keyword_rank.get(doc_id),
        }
        results.append(item)
    return results




def main():
    if not check_ollama():
        sys.exit(1)

    client = chromadb.PersistentClient(path="./chroma_db")
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url=OLLAMA_URL,
        model_name=EMBED_MODEL,
    )

    collection = step1_prepare_and_load(client, ollama_ef)
    step2_metadata_filter(collection)
    step3_keyword_filter(collection)
    step4_why_hybrid(collection)
    step5_hybrid(collection)


if __name__ == "__main__":
    main()