import ollama


CHAT_MODEL = "qwen2.5:7b"

SYSTEM = ("당신은 문서 기반 고객지원 어시스턴트입니다. 아래 [문맥]에 실제로 있는 내용만 근거로 답하세요. "
          "[문맥]에 답의 근거가 없으면 반드시 정확히 '문서에서 찾을 수 없습니다.'라고만 답하고 절대 지어내지 마세요. "
          "답에 근거로 쓴 문맥 번호를 [1] 처럼 표시하세요.")


def split_sentences(text):
    sentences = []
    for part in text.split(". "):
        part = part.strip()
        if part:
            if not part.endswith("."):
                part = part + "."
            sentences.append(part)
    return sentences


def recursive_split(text, max_size):
    sentences = split_sentences(text)
    chunks = []
    current = ""
    for s in sentences:
        if current and len(current) + 1 + len(s) > max_size:
            chunks.append(current)
            current = s
        elif current:
            current = current + " " + s
        else:
            current = s
    if current:
        chunks.append(current)
    return chunks


def ingest(collection, doc, max_size):
    chunks = recursive_split(doc, max_size)
    ids = []
    for i in range(len(chunks)):
        ids.append("c%d" % i)
    collection.add(ids=ids, documents=chunks)
    return chunks


def retrieve(collection, question, k):
    result = collection.query(query_texts=[question], n_results=k)
    ids = result["ids"][0]
    docs = result["documents"][0]
    return ids, docs


def build_prompt(context_docs, question):
    lines = ["[문맥]"]
    for i in range(len(context_docs)):
        lines.append("[%d] %s" % (i + 1, context_docs[i]))
    lines.append("")
    lines.append("[질문] %s" % question)
    lines.append("답변:")
    return "\n".join(lines)


def generate(system, user):
    res = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0},
    )
    return res["message"]["content"]


def answer(collection, question, k):
    ids, docs = retrieve(collection, question, k)
    user = build_prompt(docs, question)
    text = generate(SYSTEM, user)
    sources = []
    for i in range(len(ids)):
        sources.append((ids[i], docs[i]))
    return {"answer": text, "sources": sources}