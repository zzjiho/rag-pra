RRF_K = 60


def vector_search(col, query, k):
    result = col.query(query_texts=[query], n_results=k)
    return result["ids"][0]


def hybrid_search(col, query, k):
    # (a) 벡터 순위: 벡터 검색 상위 k개에 1위부터 순위를 매긴다.
    vector_result = col.query(query_texts=[query], n_results=k)
    vector_ids = vector_result["ids"][0]

    vector_rank = {}
    for i in range(len(vector_ids)):
        vector_rank[vector_ids[i]] = i + 1

    # (b) 키워드 순위: 질문 단어가 본문에 부분문자열로 몇 개 겹치는지 세고, 0점은 제외한다.
    all_docs = col.get()
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

    keyword_candidates = []
    for i in range(len(all_ids)):
        doc_id = all_ids[i]
        if keyword_score[doc_id] > 0:
            keyword_candidates.append(doc_id)

    keyword_candidates.sort(key=lambda doc_id: keyword_score[doc_id], reverse=True)

    keyword_rank = {}
    for i in range(len(keyword_candidates)):
        keyword_rank[keyword_candidates[i]] = i + 1

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

    return ranked_ids[:k]
