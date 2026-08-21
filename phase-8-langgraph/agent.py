from typing import TypedDict

import ollama
from langgraph.graph import StateGraph, START, END


CHAT_MODEL = "qwen2.5:7b"
TOP_K = 3
MAX_TRIES = 2


GRADE_SYSTEM = ("너는 검색 결과 채점기다. 아래 [문맥]이 [질문]에 답할 근거를 담고 있으면 정확히 'yes', "
                "담고 있지 않으면 정확히 'no' 한 단어로만 답하라. 다른 말은 하지 마라.")

TRANSFORM_SYSTEM = ("너는 검색 질의 재작성기다. 아래 질문을 벡터 검색이 더 잘 되도록 핵심 키워드를 살려 "
                    "한 문장으로 다시 써라. 재작성한 질문 한 문장만 출력하라.")

GENERATE_SYSTEM = ("아래 [문맥]에 있는 내용만 근거로 답하라. "
                   "[문맥]에 근거가 없으면 정확히 '문서에서 찾을 수 없습니다.'라고만 답하라. "
                   "답에 근거로 쓴 문맥 번호를 [1] 처럼 표시하라.")


# --- 인제스트용 청킹 (Phase 5 로직) ---

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


# --- 채팅 헬퍼 ---

def _chat(system, user):
    res = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0},
    )
    return res["message"]["content"]


def _build_context(docs):
    lines = ["[문맥]"]
    for i in range(len(docs)):
        lines.append("[%d] %s" % (i + 1, docs[i]))
    return "\n".join(lines)


# --- 상태 ---

class State(TypedDict):
    question: str
    documents: list
    relevant: str
    generation: str
    tries: int


# --- 노드 함수 ---
# retrieve는 collection이 필요해서 build_graph 안에서 collection을 캡처한 내부 함수로 만든다.

def grade(state):
    docs = state["documents"]
    question = state["question"]
    user = _build_context(docs) + "\n\n[질문] %s" % question
    answer = _chat(GRADE_SYSTEM, user)
    if "yes" in answer.lower():
        relevant = "yes"
    else:
        relevant = "no"
    print("[grade] 관련성=%s" % relevant)
    return {"relevant": relevant}


def transform_query(state):
    question = state["question"]
    new_question = _chat(TRANSFORM_SYSTEM, question).strip()
    print("[transform] 재작성 -> %s" % new_question)
    return {"question": new_question}


def generate(state):
    docs = state["documents"]
    question = state["question"]
    user = _build_context(docs) + "\n\n[질문] %s\n답변:" % question
    text = _chat(GENERATE_SYSTEM, user).strip()
    print("[generate] 답변 생성")
    return {"generation": text}


def route(state):
    if state["relevant"] == "yes":
        print("[route] 관련 있음 -> generate")
        return "generate"
    if state["tries"] >= MAX_TRIES:
        print("[route] 관련 없음 + 재시도 %d회 -> 포기하고 generate" % state["tries"])
        return "generate"
    print("[route] 관련 없음 -> transform_query (재검색)")
    return "transform_query"


# --- 그래프 빌더 ---

def build_graph(collection):
    def retrieve(state):
        question = state["question"]
        result = collection.query(query_texts=[question], n_results=TOP_K)
        docs = result["documents"][0]
        tries = state.get("tries", 0) + 1
        print("[retrieve] 검색(시도 %d) -> 청크 %d개" % (tries, len(docs)))
        return {"documents": docs, "tries": tries}

    builder = StateGraph(State)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade", grade)
    builder.add_node("transform_query", transform_query)
    builder.add_node("generate", generate)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        route,
        {"generate": "generate", "transform_query": "transform_query"},
    )
    builder.add_edge("transform_query", "retrieve")
    builder.add_edge("generate", END)

    return builder.compile()
