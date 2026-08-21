# Phase 8 — LangGraph 통합 (Agentic RAG)

> Phase 0부터 7까지 우리가 만든 RAG는 전부 **직선**이었다. 질문이 오면 `retrieve`로 청크를 뽑고,
> `generate`로 답을 쓴다. 화살표가 한 방향으로만 흐른다 — 검색 → 생성 → 끝. Phase 7에서 검색을
> 세 각도로 보강했어도 흐름 자체는 여전히 직선이었다. 뽑은 문서가 좋든 나쁘든 파이프라인은 무조건
> 다음 칸으로 넘어가 답을 만든다.
>
> 그런데 실무에서 사람이 검색을 쓸 때는 그렇게 안 한다. 검색 결과를 보고 **"이게 쓸 만한가?"** 를
> 먼저 판단한다. 쓸 만하면 그걸로 답하고, 엉뚱하면 검색어를 바꿔 **다시 찾고**, 그래도 없으면
> **"모르겠다"** 고 인정한다. 이 "보고 → 판단 → 분기 → (필요하면) 되돌아가기"가 직선 파이프라인에는
> 없다. `if`문 몇 개로 흉내 낼 수는 있지만, 분기가 늘고 루프가 생기고 상태를 여러 노드가 나눠 갱신하기
> 시작하면 금세 얽힌다.
>
> 그 분기·루프·상태 관리를 **그래프**로 다루는 도구가 **LangGraph**다. 노드(하는 일)와 엣지(다음
> 어디로)를 선언하면, 조건부 엣지로 갈림길을 만들고 엣지를 되돌려 루프를 만들 수 있다. 이 위에
> Phase 0~7의 검색·생성을 노드로 얹으면, **검색 결과를 스스로 평가해 재검색할지 답할지 정하는**
> 에이전트가 된다. 이걸 **Agentic RAG**라 부르고, 우리가 만들 건 그중 한 형태인 **self-corrective RAG**다.
> 이 커리큘럼의 종착점이다 — 지금까지 만든 부품이 전부 이 그래프 안으로 들어간다.

**이 장을 마치면 이런 문장을 스스로 설명할 수 있다.**
- "직선 RAG"와 "에이전트 RAG"의 차이는 무엇인가? 직선으로는 안 되고 그래프가 필요한 이유는?
- LangGraph의 네 부품 — State(TypedDict) · 노드(state→부분상태) · 엣지 · 조건부 엣지 — 은 각각 무엇을 맡나?
- 노드 함수는 왜 **전체 상태가 아니라 바꾼 키만** 담은 dict를 반환하나? 반환하면 무슨 일이 일어나나?
- 조건부 엣지의 라우터(`route`)는 무엇을 반환하고, 그 반환값이 `add_conditional_edges`의 매핑과 어떻게 연결되나?
- Phase 0~7의 검색·생성이 이 그래프의 어느 노드가 되나?
- `grade` 노드는 무엇을 판정하고, 그 결과로 `route`가 어떻게 분기하나(yes → ? / no → ? / 재시도 한도 → ?)?
- 환불 질문과 주차장 질문은 그래프를 각각 어떤 경로로 통과하나? 왜 하나는 바로 답하고 하나는 재검색 후 포기하나?
- 무한 루프는 왜 위험하고, `MAX_TRIES`가 그걸 어떻게 막나?
- "에이전트의 장기 기억"이 결국 우리 `retrieve` 노드와 같은 메커니즘이라는 말은 무슨 뜻인가?
- LangGraph checkpointer는 무엇을 지속시키나(대화 상태), 우리 코드는 왜 그걸 안 쓰나?

---

## 1. 복습 — 왜 LangGraph인가 (직선 RAG의 한계)

Phase 6에서 완성한 RAG 루프를 한 줄로 쓰면 이렇다.

```
질문 → retrieve(검색) → generate(생성) → 답
```

화살표가 한 방향이다. 이 구조에는 판단이 없다. `retrieve`가 무엇을 뽑아 오든 `generate`는 그걸 근거로
답을 만든다. 뽑힌 청크가 질문과 상관없어도 파이프라인은 멈추지 않는다 — 그냥 그 엉뚱한 근거로 성실히
답하거나(할루시네이션의 온상), Phase 6에서 심어 둔 "근거 없으면 못 찾겠다고 답하라"는 지시 덕에 거부한다.
어느 쪽이든 **한 번 검색하고 끝**이다.

실무 에이전트가 하길 바라는 건 그 이상이다. 세 가지 판단이 더 필요하다.

- **검색 결과가 쓸 만한가?** — 뽑아 온 청크가 진짜 질문에 답할 근거를 담았는지 확인한다.
- **다시 검색할까?** — 아니라면 검색어를 바꿔서 한 번 더 시도한다.
- **못 찾으면?** — 몇 번 시도해도 없으면 포기하고 "모르겠다"고 인정한다. (무한정 재시도하지 않는다.)

이 세 판단을 넣으면 흐름이 더는 직선이 아니다. **갈림길**(쓸 만하면 답하고 아니면 재검색)과
**되돌아가기**(재검색은 검색 단계로 다시 감)가 생긴다.

```
질문 → retrieve → grade(쓸 만한가?) ─ yes ─→ generate → 답
                        │
                        no
                        ↓
                  transform_query(질문 고쳐 쓰기) ──┐
                        ↑                          │
                        └──────── retrieve로 되돌아감 (루프)
```

이걸 `if`/`while`로 짤 수도 있다. 실제로 노드가 넷이면 그럭저럭 짜진다. 문제는 이런 그래프가 실무에서
금세 커진다는 것이다 — 노드가 여러 개, 갈림길이 여러 개, 어떤 노드는 상태의 이 키를 갱신하고 저 노드는
저 키를 갱신하고, 루프도 여러 겹. 이걸 손으로 짠 제어문으로 관리하면 "지금 흐름이 어디 있고 상태가
어떤지"가 코드에 흩어져 안 보인다.

**LangGraph는 이 흐름을 데이터로 선언하게 한다.** 노드는 함수로, 엣지는 "A 다음 B"로, 갈림길은
"이 노드 다음엔 라우터가 정한 곳으로"라고 적으면, 프레임워크가 상태를 노드 사이로 실어 나르며 그래프를
돌린다. 흐름이 **그림 한 장으로 보인다** — 어떤 노드가 있고, 어디서 갈라지고, 어디로 되돌아가는지.
분기하고 루프 도는 RAG를 다루기에 알맞은 그릇이다.

> LangGraph는 LangChain 팀이 만든 별개 라이브러리다. Phase 0~7은 LangChain 없이 `chromadb`·`ollama`를
> 직접 썼고, 이번에도 검색·LLM 호출은 그대로 직접 한다. LangGraph는 그 위에서 **흐름(제어)만** 맡는다 —
> 검색이나 생성을 대신 해 주는 게 아니라, 우리가 만든 노드들을 어떤 순서로 어떤 조건에 돌릴지를 관리한다.

---

## 2. LangGraph 기본 — 네 부품

LangGraph로 그래프 하나를 만드는 데 필요한 건 네 가지다. State, 노드, 엣지, 조건부 엣지. 하나씩 본다.

### ① State — 노드들이 공유하는 데이터

그래프가 도는 동안 노드들이 함께 읽고 쓰는 데이터 묶음이다. `TypedDict`로 어떤 키가 있는지 선언한다.
우리 에이전트의 상태는 `agent.py`에 이렇게 있다.

```python
from typing import TypedDict

class State(TypedDict):
    question: str      # 지금 검색에 쓸 질문 (transform_query가 갱신할 수 있음)
    documents: list    # retrieve가 뽑아 온 청크들
    relevant: str      # grade의 판정 결과 "yes"/"no"
    generation: str    # generate가 만든 최종 답
    tries: int         # retrieve를 몇 번 돌았나 (무한 루프 방지용)
```

State는 그래프가 도는 내내 흘러 다니는 **하나의 dict**라고 보면 된다. `invoke`할 때 초기 dict를 주면,
노드들이 차례로 이 dict를 갱신해 가고, 끝나면 최종 dict가 나온다.

### ② 노드 — `state`를 받아 **바꾼 키만** 반환하는 함수

노드는 평범한 파이썬 함수다. `state`(현재 상태 dict)를 받아서, **자기가 바꾼 키만** 담은 부분 dict를
반환한다. 전체 상태를 다시 만들어 반환하는 게 아니다.

```python
def grade(state):
    docs = state["documents"]          # 필요한 키를 읽고
    question = state["question"]
    ...
    return {"relevant": relevant}      # 바꾼 키만 반환 -> 이 키만 갱신됨
```

`grade`는 `relevant` 한 키만 돌려준다. 그러면 LangGraph가 상태의 `relevant`만 그 값으로 갈아 끼우고,
`documents`·`question` 같은 나머지 키는 그대로 둔다. **부분 갱신(partial update)** 이라 부른다. 노드마다
자기 담당 키만 손대면 되니, 상태가 커져도 각 노드는 자기 일만 신경 쓴다.

### ③ 엣지 — "이 노드 다음은 저 노드"

노드를 잇는 선이다. `add_edge("a", "b")`는 "a가 끝나면 b로 가라"는 뜻. 특별한 두 지점이 있다.

- `START` — 그래프의 입구. `add_edge(START, "retrieve")`는 "시작하면 retrieve부터"라는 뜻.
- `END` — 그래프의 출구. `add_edge("generate", END)`는 "generate가 끝나면 그래프 종료"라는 뜻.

### ④ 조건부 엣지 — 갈림길

엣지가 고정된 한 방향이라면, 조건부 엣지는 **라우터 함수가 그때그때 다음 노드를 정하는** 갈림길이다.

```python
builder.add_conditional_edges(
    "grade",              # 이 노드가 끝난 다음에
    route,                # 이 라우터 함수를 불러서
    {"generate": "generate", "transform_query": "transform_query"},  # 반환값 -> 갈 노드
)
```

라우터 `route(state)`는 상태를 보고 **문자열 키 하나를 반환**한다. 그 문자열을 위 매핑에서 찾아, 대응하는
노드로 간다. `route`가 `"generate"`를 반환하면 `generate` 노드로, `"transform_query"`를 반환하면
`transform_query` 노드로. 노드 함수(부분 상태 dict를 반환)와 라우터 함수(문자열 키를 반환)는 반환 타입이
다르다는 점을 기억하면 된다.

### 조립 — compile / invoke

부품을 `StateGraph`에 등록하고, `compile()`하면 실행 가능한 그래프가 나온다. `invoke(초기상태)`로 돌린다.

아래는 개념만 보여 주는 최소 예시다(우리 에이전트가 아니라 순수 LangGraph 데모). **0에서 시작해 3이 될
때까지 1씩 더하는** 그래프 — 노드 하나가 자기 자신으로 되돌아가는 루프다. Agentic RAG의 "조건 맞을 때까지
되돌아가기" 구조가 바로 이 모양이다.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    count: int

def step(state):
    n = state["count"] + 1
    print("step ->", n)
    return {"count": n}          # 바꾼 키만 반환

def route(state):
    if state["count"] >= 3:
        return "done"            # 3 이상이면 끝
    return "again"               # 아니면 다시 step

builder = StateGraph(State)
builder.add_node("step", step)
builder.add_edge(START, "step")
builder.add_conditional_edges("step", route, {"again": "step", "done": END})
graph = builder.compile()

out = graph.invoke({"count": 0})
print("최종:", out["count"])     # 3
```

돌리면 `step`이 세 번 찍히고 최종 3이 나온다. `step`은 `count`만 갱신하고, `route`는 `count`를 보고
`"again"`(자기 자신으로 루프) 또는 `"done"`(END)을 반환한다. 이 작은 예시에 Agentic RAG의 골격이 다
들어 있다 — **노드가 상태를 갱신하고, 라우터가 상태를 보고 계속할지 끝낼지 정하고, 조건이 되돌아가기와
끝내기를 가른다.** 우리 에이전트는 여기서 `count`가 `relevant`/`tries`로, `"again"`이 `transform_query`로,
`"done"`이 `generate`로 바뀐 것뿐이다.

---

## 3. RAG를 노드로 — Phase 0~7이 그래프 안으로

이제 부품을 우리 것으로 바꾼다. Phase 0~7에서 만든 검색과 생성이 그대로 **노드**가 된다. `agent.py`의
노드 넷을 보면, 하는 일은 지난 Phase에서 다 해 본 것들이다.

| 노드 | 하는 일 | 어디서 왔나 | 반환하는 키 |
| --- | --- | --- | --- |
| `retrieve` | 질문으로 벡터 DB에서 가까운 청크 top-k를 뽑는다 | Phase 1~5 (임베딩·ChromaDB·청킹) | `documents`, `tries` |
| `grade` | 뽑힌 청크가 질문에 답할 근거를 담았는지 LLM으로 yes/no 판정 | **새로 등장** (에이전트의 판단) | `relevant` |
| `transform_query` | 근거가 없으면 질문을 검색 잘 되게 다시 쓴다 | Phase 7 Multi-Query의 질의 재작성 아이디어 | `question` |
| `generate` | 문맥만 근거로 답하고, 근거 없으면 못 찾겠다고 답한다 | Phase 6 (생성 + grounding + 인용) | `generation` |

`retrieve`는 `collection`이 있어야 검색을 하므로, `build_graph(collection)` 안에서 `collection`을 캡처한
내부 함수로 만든다. 나머지 셋은 컬렉션이 필요 없어 모듈 레벨 함수다.

```python
def build_graph(collection):
    def retrieve(state):
        question = state["question"]
        result = collection.query(query_texts=[question], n_results=TOP_K)
        docs = result["documents"][0]         # [[...]] 한 겹 벗기기
        tries = state.get("tries", 0) + 1     # 검색할 때마다 +1
        return {"documents": docs, "tries": tries}
    ...
```

`retrieve`가 반환하는 `tries`에 주목한다. 이 노드는 검색할 때마다 시도 횟수를 1 올린다. 이 숫자가
나중에 "몇 번이나 재검색했나"를 세는 근거가 되고, 무한 루프를 막는 열쇠가 된다(§4).

`grade`·`transform_query`·`generate`는 전부 LLM을 부른다. 세 노드가 쓰는 프롬프트는 `agent.py` 위쪽에
상수로 있다.

- `GRADE_SYSTEM` — "문맥이 질문에 답할 근거를 담았으면 정확히 'yes', 아니면 'no' 한 단어로만 답하라."
- `TRANSFORM_SYSTEM` — "질문을 벡터 검색이 더 잘 되도록 핵심 키워드를 살려 한 문장으로 다시 써라."
- `GENERATE_SYSTEM` — "문맥에 있는 내용만 근거로 답하라. 근거 없으면 '문서에서 찾을 수 없습니다.'라고만.
  근거로 쓴 문맥 번호를 [1]처럼 표시하라." (Phase 6의 grounding 프롬프트 그대로다.)

LLM 호출은 Phase 6에서 쓰던 `ollama.chat(...)`을 얇게 감싼 `_chat(system, user)` 헬퍼로 한다.
`temperature=0`으로 부른다 — 같은 입력에 같은 판정이 나오도록.

핵심은 이거다. **새로 배운 건 노드를 잇는 흐름(그래프)뿐이고, 노드 안에서 하는 일은 Phase 0~7의 반복이다.**
검색은 Phase 1~5, 생성·grounding·인용은 Phase 6, 질의 재작성은 Phase 7. `grade` 하나만 새 얼굴이다.

---

## 4. Agentic RAG — 검색 결과를 스스로 평가하고 되돌아간다

노드 넷을 엣지로 이으면 그래프가 완성된다. `build_graph`의 엣지 선언이 §1에서 그렸던 그림 그대로다.

```python
builder.add_edge(START, "retrieve")            # 시작 -> 검색
builder.add_edge("retrieve", "grade")          # 검색 -> 채점
builder.add_conditional_edges(                 # 채점 -> 갈림길
    "grade", route,
    {"generate": "generate", "transform_query": "transform_query"},
)
builder.add_edge("transform_query", "retrieve")  # 재작성 -> 다시 검색 (루프!)
builder.add_edge("generate", END)              # 생성 -> 끝
```

`transform_query`에서 `retrieve`로 되돌아가는 엣지가 **루프**다. 근거를 못 찾으면 질문을 고쳐 검색
단계로 다시 돌아간다. 갈림길에서 어디로 갈지는 라우터 `route`가 정한다.

```python
def route(state):
    if state["relevant"] == "yes":
        return "generate"                # 근거 있음 -> 바로 답
    if state["tries"] >= MAX_TRIES:
        return "generate"                # 근거 없지만 너무 많이 시도함 -> 포기하고 답(못 찾겠다)
    return "transform_query"             # 근거 없음 -> 질문 고쳐 재검색
```

세 갈래이고 위에서부터 순서대로 검사한다. 핵심은 **재검색(`transform_query`)이 맨 마지막 기본값**이라는
점이다 — 재시도 한도 검사(`tries >= MAX_TRIES`)보다 뒤에 있어야, 한도에 도달했을 때 재검색 대신
포기(`generate`)로 빠진다. (앞의 두 검사는 둘 다 `generate`로 가므로 yes 검사와 한도 검사의 선후는
결과를 바꾸지 않는다. 무한 루프를 막는 건 재검색이 한도 검사 뒤에 온다는 것 하나다.)

### 무한 루프 방지 — `MAX_TRIES`

`grade`가 계속 `no`만 내면 `transform_query → retrieve → grade → transform_query → ...`가 영원히 돈다.
LLM이 재작성한 질문으로도 끝내 근거를 못 찾는 경우다. 이걸 막는 게 `route` 두 번째 줄이다.
`retrieve`가 돌 때마다 `tries`가 1씩 오르므로, `tries >= MAX_TRIES`(=2)가 되면 라우터가 재검색 대신
`generate`로 보낸다. 그럼 `generate`가 (근거 없는) 문맥을 받아 `GENERATE_SYSTEM` 지시대로
"문서에서 찾을 수 없습니다."라고 답하고 끝난다. **못 찾으면 포기하고 인정하는** 출구다.

에이전트에 루프를 넣을 때는 반드시 이런 종료 조건이 있어야 한다. 판단하는 주체가 LLM이라 "언젠간
찾겠지"를 스스로 멈추지 못할 수 있기 때문이다. 카운터 하나(`tries`)와 상한(`MAX_TRIES`)이 안전장치다.

### 실측 흐름 — 두 질문이 그래프를 통과하는 경로

`documents.py`의 질문 둘은 일부러 서로 다른 경로를 타도록 골랐다.

**질문 1 — "단순 변심 환불하면 배송비 얼마?"** (문맥에 근거 있음)

```
retrieve(tries=1) → grade → relevant=yes → route:"generate" → generate → END
```

첫 검색에서 환불 배송비 청크가 잡히고, `grade`가 근거 있다고 `yes`. 라우터가 바로 `generate`로 보내
"3000원..."을 답한다. 재검색 없이 곧장 끝. `tries=1`.

**질문 2 — "주차장 있나요?"** (문맥에 없음)

```
retrieve(tries=1) → grade → relevant=no → route:"transform_query"
       → transform_query(질문 재작성) → retrieve(tries=2) → grade → relevant=no
       → route: tries>=2 이므로 "generate" → generate → END
```

코퍼스에 주차장 얘기가 없으니 `grade`가 `no`. 라우터가 `transform_query`로 보내 질문을 고쳐 다시
검색하지만, 없는 정보는 재작성해도 안 나온다. 두 번째 `grade`도 `no`. 이번엔 `tries=2 >= MAX_TRIES`라
라우터가 재검색을 포기하고 `generate`로 보낸다. `generate`는 "문서에서 찾을 수 없습니다."라고 답하고 끝.
`tries=2`.

이게 핵심 대비다. **같은 그래프인데 질문에 따라 경로가 다르다.** 첫 질문은 지름길(retrieve→grade→generate),
둘째 질문은 재검색 한 바퀴를 돌고 한도에서 멈춘다. **경로를 정하는 건 우리가 짠 `if`가 아니라, 검색
결과를 보고 판단한 에이전트 자신**이다. 직선 RAG에는 없던 것이다.

> **작은 코퍼스의 정직한 한계.** 이 코퍼스는 문장 5개가 청크 3개로 쪼개지고(`recursive_split`,
> `max=90`), `TOP_K=3`이라 `retrieve`는 매번 **청크 3개 전부**를 돌려준다. 그래서 주차장 질문의
> 재검색은 사실 새 문서를 찾아 주지 못한다 — 같은 3개가 또 나올 뿐이다. 이 데모가 보여 주는 건
> "재작성하면 없던 답이 생긴다"가 아니라 **제어 흐름** 이다: 에이전트가 검색 결과를 평가하고(grade),
> 아니라고 판단하면 재검색을 시도하고(transform_query→retrieve), 그래도 안 되면 한도에서 포기한다.
> 문서가 수만 개인 실무에서는 재작성한 질의가 실제로 다른 청크를 끌어와 두 번째 검색이 답을 살리는
> 일이 생긴다. 여기서 확인할 것은 **분기와 루프와 종료가 상태(relevant·tries)에 따라 자동으로 도는
> 동작**이다.

---

## 5. 장기 기억 = 벡터 검색 (개념)

에이전트 이야기를 하면 꼭 "기억(memory)"이 나온다. 에이전트가 과거를 기억한다는 게 특별한 새 기술처럼
들리지만, **장기 기억(long-term memory)의 실체는 대개 우리가 이미 만든 것**이다.

에이전트의 장기 기억이란 결국 **어딘가에 저장해 둔 정보를 필요할 때 검색해 오는 것**이다. 그 "어딘가"가
보통 벡터 스토어다. 과거 대화, 사용자가 알려 준 선호, 에이전트가 알아낸 사실을 임베딩해 벡터 DB에 넣어
두고, 나중에 관련될 때 질문으로 검색해 꺼낸다. **이 꺼내는 동작이 정확히 우리 `retrieve` 노드다.** 질문을
임베딩해 가까운 걸 뽑는 그 메커니즘.

그러니 이렇게 정리된다.

- 우리 `DOC` 코퍼스는 **읽기 전용 지식 베이스**다. 미리 넣어 두고 검색만 한다.
- 에이전트의 장기 기억은 거기에 **쓰기**가 더해진 것이다 — 새로 알게 된 사실을 같은 벡터 스토어에
  넣고, 다음에 검색해 꺼낸다. **읽는 경로(retrieve)는 완전히 똑같다.**

즉 "에이전트에 기억을 달자"는 대부분 "벡터 스토어를 하나 두고 읽고 쓰자"는 말이다. Phase 1~7에서 벡터
검색을 그렇게 오래 다진 게 여기서 또 쓰인다. 기억은 새 개념이 아니라 **검색의 응용**이다.

### LangGraph checkpointer — 대화 상태의 지속 (개념)

기억에는 결이 다른 하나가 더 있다. 방금 얘기한 장기 기억이 "지식을 벡터로 저장·검색"이라면,
**checkpointer**는 "그래프의 **State**를 대화 턴 사이에 유지"하는 장치다. 짧게 짚고 넘어간다.

우리 그래프는 `invoke`할 때마다 초기 상태를 새로 받고, 끝나면 그 상태가 사라진다. `main.py`에서 질문
둘을 각각 `graph.invoke(initial)`로 부르는데, 두 호출은 서로 아무것도 공유하지 않는다 — 매번 백지에서
시작한다. 단발 질의응답이라 이걸로 충분하다.

그런데 여러 턴 이어지는 대화라면 이전 턴의 상태(예: 앞서 나눈 대화, 이전 답)를 다음 턴이 이어받아야
한다. LangGraph의 checkpointer는 `compile`할 때 붙이는 저장소로, 각 대화(스레드 id로 구분)의 State를
저장해 뒀다가 같은 스레드로 다시 `invoke`하면 이어서 돌게 해 준다. 대략 이런 모양이다(우리 코드엔 없다,
개념만).

```python
# 개념 예시 — 이 파일에 있는 코드가 아니다.
graph = builder.compile(checkpointer=some_saver)
graph.invoke(state, config={"configurable": {"thread_id": "user-42"}})  # 이 스레드 상태를 저장·복원
```

정리하면 두 종류의 "기억"이 있다.

| 종류 | 무엇을 유지하나 | 실체 | 우리 코드 |
| --- | --- | --- | --- |
| 장기 기억 | 지식·사실 | 벡터 스토어(= `retrieve`의 대상) | `retrieve` 노드가 그 읽기 경로 |
| checkpointer | 그래프 State(대화 진행 상태) | 스레드별 상태 저장소 | 안 씀(단발 invoke) |

둘 다 이 데모에서는 코드로 구현하지 않았다. 단발 질의응답에는 필요 없기 때문이다. 다만 실무에서
에이전트를 대화형으로 키울 때 어디를 손대는지는 알아 두면 된다 — 지식을 늘리려면 벡터 스토어(우리
`retrieve`가 보는 그것), 대화 맥락을 이으려면 checkpointer.

> LangGraph 공식 문서의 용어와 맞춰 두면 헷갈리지 않는다. 문서는 checkpointer를
> **short-term memory**(스레드 범위의 State 지속)라 부르고, 스레드를 넘나드는
> **long-term memory**는 별도의 **Store** API로 다룬다. 이 Store가 담는 기억의 실체가 바로 벡터
> 검색이다. 위에서 "장기 기억 = `retrieve`"라고 한 건 LangGraph의 Store와 경쟁하는 다른 개념이
> 아니라, 그 Store 밑에서 도는 검색 메커니즘을 가리킨 것이다.

---

## 6. 왜 이게 직무의 핵심인가 — 커리큘럼 회고

이 커리큘럼의 목표는 처음부터 하나였다. **LangGraph로 판단하는 AI 에이전트를 만드는 것.** Phase 8이
그 종착점이고, 지금 만든 그래프가 그 목표의 축소판이다.

돌아보면 Phase 0~7은 전부 이 그래프의 **노드 안**으로 들어갔다.

| Phase | 배운 것 | 이 그래프에서 |
| --- | --- | --- |
| 0 | 임베딩 — 텍스트를 벡터로 | `retrieve`가 질문을 벡터로 만들어, 인제스트 때 벡터화해 둔 문서들과 비교 |
| 1 | ChromaDB — 벡터 저장·검색 | `retrieve`가 `collection.query`로 꺼냄 |
| 2 | 미니 검색기 — 검색 파이프라인 | `retrieve` 노드 전체 |
| 3 | 필터·하이브리드 — 검색 정밀화 | `retrieve`의 품질을 올리는 도구들 |
| 4 | 평가 — recall·MRR로 검색 재기 | 어떤 검색·청킹이 나은지 판단하는 잣대 |
| 5 | 청킹 — 문서를 조각으로 | `recursive_split`이 인제스트에서 그대로 |
| 6 | 생성·grounding·인용 — R에 G를 붙임 | `generate` 노드 전체 |
| 7 | 고급 검색 — Multi-Query·HyDE·재랭킹 | 질의 재작성이 `transform_query`로, 검색 보강 일반 |

새로 배운 건 **노드를 잇는 방식**뿐이다. 검색도 생성도 다 만들어 본 것이고, LangGraph는 그것들을
**언제 어떤 순서로 어떤 조건에 돌릴지** 정하는 뼈대를 얹었다. 여기에 딱 하나 새 판단 노드(`grade`)를
더해, 파이프라인이 **직선에서 그래프로** 바뀌었다.

이게 왜 직무의 핵심인가. 실무에서 "AI 에이전트를 만든다"는 건 대부분 이런 그래프를 설계하는 일이다.

- 무슨 노드가 필요한가 (검색? 도구 호출? 판단? 생성?)
- 어디서 갈라지는가 (검색이 실패하면? 도구가 에러를 내면? 사용자가 확인을 요구하면?)
- 무엇을 상태에 담아 노드 사이로 넘길 것인가
- 루프는 어디서 돌고, 무엇으로 멈추는가 (`MAX_TRIES` 같은 안전장치)

우리가 만든 self-corrective RAG는 그 설계의 한 사례일 뿐이다. 검색 대신 도구를 부르고, `grade` 대신
다른 판단을 넣고, 노드를 더 붙이면 다른 에이전트가 된다. 골격은 같다 — **상태를 들고 노드를 돌며,
결과를 보고 다음을 판단해 분기·루프하는 그래프.** 그 골격을 이번에 손으로 만들어 봤다.

---

## 7. 파일 구조 · 설치 · 실행 · 그래프 지도 · 체크리스트

### 파일 구조

| 파일 | 역할 |
| --- | --- |
| `documents.py` | **데이터.** `DOC`(고객센터 정책 문단)와 `QUESTIONS`(경로가 갈리는 질문 2개). 로직 없음. |
| `agent.py` | **에이전트 코어.** `State`(TypedDict) · 노드 넷(`retrieve`/`grade`/`transform_query`/`generate`) · 라우터 `route` · 그래프 조립 `build_graph` · 인제스트용 청킹(`recursive_split`, Phase 5). 프롬프트 상수도 여기. |
| `main.py` | **실행.** Ollama·모델 점검 → 컬렉션 준비·인제스트 → 그래프 조립 → 질문마다 `invoke` → 흐름과 최종 답을 화면에 펼침. |
| `requirements.txt` | `chromadb`, `ollama`, **`langgraph`**(이번에 추가). |

### 설치

이 폴더(`phase-8-langgraph`)에서:

```bash
python3 -m venv .venv
source .venv/bin/activate           # 새 터미널을 열 때마다 이 줄로 활성화
pip install -r requirements.txt     # chromadb, ollama, langgraph
```

`langgraph`가 이번에 새로 들어간다. 나머지 둘은 앞 Phase와 같다.

### 모델 준비

임베딩 `bge-m3`와 채팅 LLM `qwen2.5:7b`는 앞 Phase에서 받았다면 그대로 쓴다. 없으면:

```bash
brew services start ollama   # 서버가 꺼져 있으면
ollama pull bge-m3           # 임베딩 (retrieve 노드)
ollama pull qwen2.5:7b       # 채팅 LLM (grade·transform_query·generate 세 노드가 이 모델을 씀)
```

`main.py`가 실행 첫머리(`check_ollama`)에서 둘 다 있는지 점검하고, 빠졌으면 어떤 `ollama pull`을 치면
되는지 알려 준다.

### 실행

```bash
python main.py
```

- 이 폴더에 `./chroma_db`가 생기고, 컬렉션(`phase8_agentic`)을 매번 지웠다 새로 만든다
  (`build_collection` 첫머리의 `delete_collection`) — 항상 깨끗한 상태에서 재현하기 위해서다.
- 세 LLM 노드는 `_chat`을 `temperature=0`으로 부른다. `grade`의 yes/no와 라우팅은 안정적으로 재현되고,
  `transform_query`의 재작성 문장과 `generate`의 답 문장은 실행마다 표현이 조금 달라질 수 있다. 확인할
  것은 **경로**다 — 환불 질문은 재검색 없이 곧장 답하고, 주차장 질문은 재검색 한 바퀴 후 한도에서 포기한다.
- 각 노드가 `print`로 자기 동작을 찍는다(`[retrieve]`, `[grade]`, `[route]`, `[transform]`, `[generate]`).
  이 로그를 순서대로 읽으면 그래프가 어느 경로로 흘렀는지 눈으로 따라갈 수 있다.

### 그래프 지도 — 노드와 엣지

```
   START → retrieve → grade ─(route)─┬─ relevant=yes ───→ generate → END
              ↑                      │                       ↑
              │                      ├─ no & tries≥MAX ───────┘  (포기하고 "못 찾겠다")
              │                      │
              └── transform_query ←──┴─ no & tries<MAX          (질문 고쳐 재검색)
```

| 노드 | 갱신 키 | 다음 |
| --- | --- | --- |
| `retrieve` | `documents`, `tries` | `grade` (고정 엣지) |
| `grade` | `relevant` | 조건부 → `route`가 결정 |
| `route`(라우터) | — (문자열 키만 반환) | `relevant`/`tries`를 보고 `generate` 또는 `transform_query` |
| `transform_query` | `question` | `retrieve` (루프 — 고정 엣지) |
| `generate` | `generation` | `END` |

### `main.py`가 보여주는 것

| 블록 | 화면 제목 | 하는 일 |
| --- | --- | --- |
| **[1]** | 그래프 구조 | 노드 넷과 조건부 라우팅, 적재된 청크 수를 요약해 보여 줌 |
| **[2]** | 실행 | 질문마다 `invoke`, 노드 로그가 실시간으로 찍히고 최종 답·`tries`를 출력 |
| **[3]** | 정리 | 두 질문의 경로 대비로 Agentic RAG의 요점을 정리 |

### Phase 8 완료 체크리스트

- [ ] "직선 RAG"(retrieve→generate 한 방향)와 "에이전트 RAG"(평가·분기·루프)의 차이를, 왜 `if`문 대신 그래프가 필요한지와 함께 설명할 수 있다.
- [ ] LangGraph 네 부품(State·노드·엣지·조건부 엣지)이 각각 무엇을 맡는지 말할 수 있다.
- [ ] 노드 함수가 **전체가 아니라 바꾼 키만** 담은 dict를 반환하고, LangGraph가 그 키만 상태에 갈아 끼운다(부분 갱신)는 걸 안다.
- [ ] 라우터 함수는 **문자열 키**를 반환하고, 그게 `add_conditional_edges` 매핑을 거쳐 다음 노드가 된다는 걸(노드 함수와 반환 타입이 다름) 안다.
- [ ] `START`/`END`가 무엇이고, `transform_query→retrieve` 엣지가 왜 루프인지 짚을 수 있다.
- [ ] 노드 넷이 각각 Phase 0~7의 무엇에서 왔는지(retrieve=1~5, generate=6, transform=7, grade=새로) 말할 수 있다.
- [ ] `grade`가 무엇을 yes/no로 판정하고, `route`가 그 결과로 어떻게 세 갈래(yes→generate / no&한도→generate / no→transform)로 분기하는지 순서까지 안다.
- [ ] 환불 질문과 주차장 질문의 경로 차이를 그래프 위에서 그려 설명할 수 있다.
- [ ] `MAX_TRIES`와 `retrieve`의 `tries` 증가가 어떻게 무한 루프를 막는지, 왜 LLM 루프에 종료 조건이 꼭 필요한지 안다.
- [ ] `TOP_K=3`·청크 3개라 재검색이 새 문서를 못 찾는데도 이 데모가 무엇을 보여 주는지(제어 흐름 자체)를 정직하게 설명할 수 있다.
- [ ] "에이전트의 장기 기억 = 벡터 스토어 검색 = `retrieve` 노드"라는 정리를, 읽기 전용 지식 베이스와 읽기+쓰기 기억의 차이와 함께 말할 수 있다.
- [ ] checkpointer가 무엇을 지속시키는지(그래프 State, 대화 턴 사이), 우리 코드가 왜 안 쓰는지(단발 invoke) 안다.
- [ ] Phase 0~7이 이 그래프의 노드 안으로 다 들어갔고, "에이전트 개발 = 이런 그래프 설계"임을 자기 말로 설명할 수 있다.
- [ ] `python main.py`를 돌려 두 질문의 노드 로그와 경로 차이를 직접 확인했다.

전부 체크됐다면, 이 커리큘럼이 목표한 자리에 도착한 것이다 — 검색과 생성을 **노드로 얹고, 검색 결과를
스스로 평가해 분기·루프하는 LangGraph 에이전트**를 손으로 만들어 봤다. 여기서부터는 노드를 바꾸고(도구
호출·다른 판단), 상태를 늘리고, 루프를 더해 원하는 에이전트로 키워 가면 된다. 골격은 이번에 익힌 그대로다.

## ref

- LangGraph 공식 문서 — StateGraph·노드·조건부 엣지·compile/invoke의 기본 개념: https://langchain-ai.github.io/langgraph/
- LangGraph, "Agentic RAG" 튜토리얼 — 검색 결과를 평가해 재검색/생성으로 분기하는 self-corrective RAG의 원형: https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/
- Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" — 검색·생성을 스스로 평가(critique)하는 아이디어의 논문 배경: https://arxiv.org/abs/2310.11511
- LangGraph, "Persistence / Checkpointer" 문서 — 대화 State를 스레드별로 지속시키는 checkpointer 개념: https://langchain-ai.github.io/langgraph/concepts/persistence/
- LangGraph, "Memory" 문서 — 에이전트의 장기 기억을 저장소(벡터 스토어 등)로 다루는 방식: https://langchain-ai.github.io/langgraph/concepts/memory/