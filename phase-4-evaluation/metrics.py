import math


def recall_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    hit = 0
    for doc_id in relevant_ids:
        if doc_id in top_k:
            hit = hit + 1
    return hit / len(relevant_ids)


def mrr(retrieved_ids, relevant_ids):
    for i in range(len(retrieved_ids)):
        rank = i + 1
        if retrieved_ids[i] in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]

    dcg = 0.0
    for i in range(len(top_k)):
        position = i + 1
        if top_k[i] in relevant_ids:
            dcg = dcg + 1.0 / math.log2(position + 1)

    ideal_hits = len(relevant_ids)
    if ideal_hits > k:
        ideal_hits = k

    idcg = 0.0
    for i in range(ideal_hits):
        position = i + 1
        idcg = idcg + 1.0 / math.log2(position + 1)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg
