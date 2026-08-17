# RAG


> `chromadb` · `ollama` · `sentence-transformers` · `FastAPI` · `LangGraph`


---

## Phase 0 — 개념 토대 & 환경

- 의미 기반 검색·임베딩이 무엇이고 왜 필요한가
- 임베딩 모델 원리 + 유사도 측정 (코사인 / L2 / 내적)
- 벡터 DB vs 기존 DB, 활용 사례
- Ollama 설치 + 임베딩 모델 실습, Python 클라이언트 셋업

**산출물:** 텍스트 → 벡터 변환 + 유사도 계산 스크립트

## Phase 1 — ChromaDB 기본기

- 영속(persistent) 모드, 컬렉션 CRUD
- Ollama 임베딩 함수 연결 + 동작 원리 분석
- 문서 CRUD (조회·생성·수정·삭제)

**산출물:** 문서를 넣고 빼는 벡터 저장소 모듈

## Phase 2 — 유사도 검색 & 첫 미니 데모

- query 검색, 거리 함수 해석 ("이 점수가 좋은 건가?")
- 청킹의 필요성
- 미니 시맨틱 검색기 제작 + 검색 진단 패턴

**산출물:** 동작하는 시맨틱 검색기 v1

## Phase 3 — 검색 품질: 필터 & 하이브리드

- HNSW 인덱스와 거리 함수 복습
- 메타데이터 설계, `where_document` 필터
- 벡터 + 키워드 하이브리드 검색

**산출물:** 필터·하이브리드 붙은 검색기 v2

## Phase 4 — 평가 ⭐

> 실무에서 가장 많이 요구되는데 놓치기 쉬운 부분.

- 골든 데이터셋 (질문 ↔ 정답 문서) 만들기
- 검색 지표: recall@k, MRR, nDCG
- 생성 지표: RAGAS (faithfulness, context precision/recall)

**산출물:** "튜닝 전후 recall@k X→Y" 개선 리포트

## Phase 5 — 청킹 심화 & 인덱스 튜닝

- Recursive / Semantic / Code / Parent-Child 청킹
- HNSW 파라미터 (`M`, `ef_construction`) 튜닝

**산출물:** 문서 종류별 청킹 전략 비교표 (Phase 4 지표로 측정)

## Phase 6 — RAG 파이프라인 & LLM 결합

- Document 인제스트 파이프라인
- LLM 결합, 프롬프트 설계, 출처 인용, 출력 품질
- FastAPI 백엔드 + 배치·페이징·성능측정
- 다른 벡터 DB (pgvector · Qdrant) 비교

**산출물:** API로 호출되는 RAG 서비스

## Phase 7 — 고급 RAG 패턴

- 재랭킹: Bi-encoder → cross-encoder, 2단계 파이프라인, 비용 균형
- Query Rewriting · HyDE · Multi-Query Retrieval
- Parent-Child Retrieval · Self-Querying · Conversational RAG

**산출물:** 고급 검색 파이프라인 v3

## Phase 8 — LangGraph 통합 ⭐

> 직무 차별화 핵심.

- RAG를 LangGraph retriever 노드로 감싸기
- Agentic RAG — 에이전트가 검색 여부·재검색을 스스로 판단
- 장기 기억(long-term memory) = 벡터 검색으로 구현

**최종 산출물:** RAG를 도구·기억으로 쓰는 LangGraph 에이전트

---
