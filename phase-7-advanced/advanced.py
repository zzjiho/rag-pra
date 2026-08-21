import re

import ollama


CHAT_MODEL = "qwen2.5:7b"


def _chat(prompt):
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return response["message"]["content"]


def embed_search(collection, query, k):
    result = collection.query(query_texts=[query], n_results=k)
    ids = result["ids"][0]
    docs = result["documents"][0]
    return ids, docs


def multi_query(question, n):
    prompt = (
        "다음 질문을 검색이 잘 되도록 서로 다른 표현으로 %d개 바꿔 써라.\n"
        "한 줄에 하나씩만 쓰고, 번호나 설명은 붙이지 마라.\n\n"
        "질문: %s"
    ) % (n, question)
    text = _chat(prompt)

    variants = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*\d+[.)]\s*", "", line)
        line = re.sub(r"^\s*[-*]\s*", "", line)
        line = line.strip()
        if line:
            variants.append(line)
    return variants


def multi_query_search(collection, question, n, k):
    variants = multi_query(question, n)

    merged_ids = []
    seen = set()
    for variant in variants:
        ids, docs = embed_search(collection, variant, k)
        for doc_id in ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            merged_ids.append(doc_id)
    return merged_ids


def hyde(question):
    prompt = (
        "다음 질문에 대한 그럴듯한 짧은 답변 문단을 상상해서 써라.\n"
        "사실 여부는 상관없다. 한 문단으로만 써라.\n\n"
        "질문: %s"
    ) % question
    return _chat(prompt).strip()


def hyde_search(collection, hypo_text, k):
    result = collection.query(query_texts=[hypo_text], n_results=k)
    ids = result["ids"][0]
    docs = result["documents"][0]
    return ids, docs


def llm_rerank(question, cand_ids, cand_docs):
    lines = []
    for i in range(len(cand_docs)):
        lines.append("%d. %s" % (i + 1, cand_docs[i]))
    candidate_block = "\n".join(lines)

    prompt = (
        "아래 후보 문서들을 질문에 관련 있는 순서대로 다시 정렬해라.\n"
        "문서 번호만 콤마로 나열하고, 설명은 절대 하지 마라.\n\n"
        "질문: %s\n\n"
        "후보:\n%s"
    ) % (question, candidate_block)
    text = _chat(prompt)

    order = re.findall(r"\d+", text)

    ranked_ids = []
    used = set()
    for token in order:
        idx = int(token) - 1
        if idx < 0 or idx >= len(cand_ids):
            continue
        if idx in used:
            continue
        used.add(idx)
        ranked_ids.append(cand_ids[idx])

    for i in range(len(cand_ids)):
        if i not in used:
            ranked_ids.append(cand_ids[i])
    return ranked_ids


def recall_at_1(ranked_ids, answer_id):
    if len(ranked_ids) == 0:
        return 0.0
    if ranked_ids[0] == answer_id:
        return 1.0
    return 0.0


def mrr(ranked_ids, answer_id):
    for i in range(len(ranked_ids)):
        rank = i + 1
        if ranked_ids[i] == answer_id:
            return 1.0 / rank
    return 0.0