import json
import sys
import urllib.error
import urllib.request

import chromadb
from chromadb.utils import embedding_functions

import chunkers
import corpus


OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "bge-m3"

FIXED_SIZE = 40
RECURSIVE_MAX = 90
SEMANTIC_DROP = 0.5
PARENT_MAX = 90


def check_ollama():
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=5) as res:
            data = json.loads(res.read())
    except (urllib.error.URLError, OSError):
        print("[준비 실패] Ollama 서버에 연결하지 못했습니다.")
        print("  서버 실행:  brew services start ollama")
        return False

    has_model = False
    for model in data.get("models", []):
        name = model.get("name", "")
        if name.startswith(EMBED_MODEL):
            has_model = True
            break

    if not has_model:
        print("[준비 실패] 임베딩 모델 '%s' 가 없습니다." % EMBED_MODEL)
        print("  모델 준비:  ollama pull %s" % EMBED_MODEL)
        return False

    print("[준비 완료] Ollama 서버 연결 OK, 임베딩 모델 '%s' 확인." % EMBED_MODEL)
    return True


def build_collection(client, ollama_ef, name, chunks, metadatas=None):
    try:
        client.delete_collection(name=name)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=name,
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )

    ids = []
    for i in range(len(chunks)):
        ids.append("c%d" % i)

    if metadatas is None:
        collection.add(ids=ids, documents=chunks)
    else:
        collection.add(ids=ids, documents=chunks, metadatas=metadatas)
    return collection


def answer_recall_at_k(col, k):
    hit_sum = 0.0
    for question, answer in corpus.GOLDEN:
        result = col.query(query_texts=[question], n_results=k)
        docs = result["documents"][0]
        hit = 0
        for d in docs:
            if answer in d:
                hit = 1
        hit_sum = hit_sum + hit
    return hit_sum / len(corpus.GOLDEN)


def print_chunks(title, chunks):
    print("  %s (총 %d개):" % (title, len(chunks)))
    for i in range(len(chunks)):
        print("    [%d] %s" % (i, chunks[i]))


def step1_compare_chunking(fixed_chunks, recursive_chunks, semantic_chunks):
    print()
    print("[1] 청킹 방식 비교 (눈으로)")
    print("  같은 문서를 세 방식으로 잘라 봅니다.")
    print()

    print_chunks("fixed_char(size=%d) — 경계 무시하고 글자 수로만 자름" % FIXED_SIZE, fixed_chunks)
    print("    -> 위 조각들을 보면 '환불할'이 '...환' / '불할...' 처럼 단어 중간에서 잘립니다.")
    print("       이렇게 잘린 조각은 그 자체로 뜻이 깨져서 LLM에 넘기기 나쁩니다.")
    print()

    print_chunks("recursive_split(max=%d) — 문장 경계를 지키며 max 이하로 모음" % RECURSIVE_MAX, recursive_chunks)
    print()

    print_chunks("semantic_split(drop=%.1f) — 인접 문장 의미가 멀어지면 끊음" % SEMANTIC_DROP, semantic_chunks)


def step2_compare_retrieval(client, ollama_ef, fixed_chunks, recursive_chunks, semantic_chunks):
    print()
    print("[2] 검색 품질 비교 (answer_recall@k)")
    print("  지표: top-k 청크 중 하나라도 정답 문자열을 포함하면 1점, 골든셋 평균.")
    print()

    fixed_col = build_collection(client, ollama_ef, "chunk_fixed", fixed_chunks)
    recursive_col = build_collection(client, ollama_ef, "chunk_recursive", recursive_chunks)
    semantic_col = build_collection(client, ollama_ef, "chunk_semantic", semantic_chunks)

    fixed_r1 = answer_recall_at_k(fixed_col, 1)
    fixed_r3 = answer_recall_at_k(fixed_col, 3)
    recursive_r1 = answer_recall_at_k(recursive_col, 1)
    recursive_r3 = answer_recall_at_k(recursive_col, 3)
    semantic_r1 = answer_recall_at_k(semantic_col, 1)
    semantic_r3 = answer_recall_at_k(semantic_col, 3)

    print("  방식             recall@1   recall@3")
    print("    %-12s   %.3f      %.3f" % ("fixed_char", fixed_r1, fixed_r3))
    print("    %-12s   %.3f      %.3f" % ("recursive", recursive_r1, recursive_r3))
    print("    %-12s   %.3f      %.3f" % ("semantic", semantic_r1, semantic_r3))

    print()
    print("  정직한 해석:")
    print("    fixed_char는 recall@1이 0.5로 확 낮습니다. 사실이 청크 경계에서 잘려")
    print("    top1 청크 하나에 정답이 온전히 담기지 못하기 때문입니다.")
    print("    recall@3에서는 fixed도 1.0이 되지만, 그건 잘린 조각이 top-3 안에")
    print("    어쨌든 나타나서일 뿐입니다. 그 조각은 '불할 경우...' 처럼 단어 중간에서")
    print("    시작하는 깨진 조각이라, LLM에 문맥으로 넘기면 품질이 나쁩니다.")
    print("    즉 top1 정확도와 조각 품질 두 가지 모두에서 naive fixed가 불리합니다.")


def step3_semantic_principle(ollama_ef):
    print()
    print("[3] Semantic 청킹 원리 — 인접 문장 코사인 유사도")
    print("  문장을 하나씩 임베딩해 바로 옆 문장과의 코사인 유사도를 봅니다.")
    print("  유사도가 낮다 = 화제가 바뀌었다 = 여기서 끊는다.")
    print()

    sentences = chunkers.split_sentences(corpus.DOC)
    embs = ollama_ef(sentences)

    for i in range(1, len(sentences)):
        sim = chunkers.cosine(embs[i - 1], embs[i])
        if sim < SEMANTIC_DROP:
            mark = "<- drop(%.1f) 아래, 여기서 끊김" % SEMANTIC_DROP
        else:
            mark = ""
        print("    문장%d -> 문장%d : 유사도 %.3f  %s" % (i - 1, i, sim, mark))

    print()
    print("  뒤쪽 두 경계에서 유사도가 0.5 밑으로 떨어져 끊깁니다.")
    print("  그래서 멤버십 문장과 고객센터 문장이 각각 따로 떨어져 나옵니다.")
    print()
    print("  주의: drop=0.5는 이 문서·이 모델(bge-m3)에서 맞춘 값일 뿐입니다.")
    print("  threshold는 데이터와 모델마다 다시 잡아야 합니다 (Phase 2 threshold 교훈).")


def step4_parent_child(client, ollama_ef):
    print()
    print("[4] Parent-Child — 작게 찾고 크게 돌려주기")
    print("  자식=문장(정밀 검색용), 부모=문단(문맥용)으로 나눕니다.")
    print()

    parents, children = chunkers.parent_child_split(corpus.DOC, PARENT_MAX)

    print("  부모(문단) 청크 %d개:" % len(parents))
    for pi in range(len(parents)):
        print("    부모[%d] %s" % (pi, parents[pi]))
    print()

    child_docs = []
    child_metas = []
    for sent, parent_index in children:
        child_docs.append(sent)
        child_metas.append({"parent": parent_index})

    child_col = build_collection(client, ollama_ef, "chunk_parent_child", child_docs, child_metas)

    question = "반품 배송비 얼마?"
    result = child_col.query(query_texts=[question], n_results=1)
    top_child = result["documents"][0][0]
    top_parent_index = result["metadatas"][0][0]["parent"]
    parent_context = parents[top_parent_index]

    print("  질문: %s" % question)
    print("    찾은 자식(정밀 매칭) : %s" % top_child)
    print("    자식의 부모 인덱스   : %d" % top_parent_index)
    print("    돌려줄 부모(문맥)    : %s" % parent_context)
    print()
    print("  교훈: 작은 자식으로 정확히 '어느 문장'인지 찾고,")
    print("        그 문장이 속한 큰 부모 문단을 문맥으로 함께 돌려줍니다.")


def step5_overlap():
    print()
    print("[5] Overlap(겹침) — 경계에서 문맥이 끊기지 않게")
    print("  새 청크를 시작할 때 이전 청크의 끝 문장을 물려주면, 경계 근처 내용이")
    print("  양쪽 청크에 모두 담겨 검색에서 놓치지 않습니다.")
    print()

    no_overlap = chunkers.recursive_split(corpus.DOC, RECURSIVE_MAX)
    with_overlap = chunkers.recursive_split_overlap(corpus.DOC, RECURSIVE_MAX, 1)

    print_chunks("recursive (overlap=0)", no_overlap)
    print()
    print_chunks("recursive (overlap=1 문장)", with_overlap)
    print()
    print("  -> overlap=1이면 인접 청크가 문장 하나씩 겹칩니다. 청크 수는 늘지만")
    print("     경계 문장이 양쪽 청크에 다 들어갑니다.")
    print("     실무에서는 LangChain의 chunk_overlap 같은 파라미터로 처리합니다.")


def main():
    if not check_ollama():
        sys.exit(1)

    client = chromadb.PersistentClient(path="./chroma_db")
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url=OLLAMA_URL,
        model_name=EMBED_MODEL,
    )

    fixed_chunks = chunkers.fixed_char(corpus.DOC, FIXED_SIZE)
    recursive_chunks = chunkers.recursive_split(corpus.DOC, RECURSIVE_MAX)
    semantic_chunks = chunkers.semantic_split(corpus.DOC, ollama_ef, SEMANTIC_DROP)

    step1_compare_chunking(fixed_chunks, recursive_chunks, semantic_chunks)
    step2_compare_retrieval(client, ollama_ef, fixed_chunks, recursive_chunks, semantic_chunks)
    step3_semantic_principle(ollama_ef)
    step4_parent_child(client, ollama_ef)
    step5_overlap()


if __name__ == "__main__":
    main()
