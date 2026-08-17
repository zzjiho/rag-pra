import json
import sys
import urllib.error
import urllib.request

import chromadb
from chromadb.utils import embedding_functions


OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "bge-m3"

DISTANCE_THRESHOLD = 0.55


DOC = """주문하신 상품은 결제가 완료되면 보통 2~3일 이내에 출고되어 배송됩니다. 도서산간 지역은 하루 이틀 더 걸릴 수 있습니다. 배송 조회는 마이페이지의 주문 내역에서 확인하실 수 있습니다.

교환은 상품에 하자가 있거나 다른 옵션을 원하실 때 신청하실 수 있습니다. 마이페이지에서 교환 신청을 누르고 사유를 선택하시면 됩니다. 회수된 상품 확인 후 새 상품이 다시 발송됩니다.

멤버십에 가입하시면 무료 배송, 포인트 적립, 전용 할인 쿠폰 등의 혜택을 받으실 수 있습니다. 등급은 최근 6개월 구매 금액에 따라 자동으로 산정됩니다. 가입과 해지는 언제든지 무료로 가능합니다.

고객센터 영업시간은 평일 오전 9시부터 오후 6시까지입니다. 점심시간인 오후 12시부터 1시까지는 상담이 어려울 수 있습니다. 주말과 공휴일은 휴무입니다.

환불은 상품을 받으신 후 7일 이내에 신청하실 수 있습니다. 신청이 승인되면 결제하신 수단으로 3영업일 이내에 환불됩니다. 단순 변심의 경우 왕복 배송비는 고객님이 부담하십니다."""


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


def split_into_paragraphs(text):
    raw_chunks = text.split("\n\n")
    chunks = []
    for chunk in raw_chunks:
        cleaned = chunk.strip()
        if cleaned:
            chunks.append(cleaned)
    return chunks


def reset_collection(client, ollama_ef, name):
    try:
        client.delete_collection(name=name)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=name,
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def step1_show_chunking(chunks):
    print()
    print("[1] 문서 & 청킹 보여주기")

    print("  원본 문서 길이: %d 글자" % len(DOC))
    print("  원본 앞부분: %s ..." % DOC[:40].replace("\n", " "))

    print("  이 한 덩어리 문서를 빈 줄 기준으로 쪼갠 결과: 청크 %d개" % len(chunks))
    for i in range(len(chunks)):
        head = chunks[i][:25]
        print("    청크 %d: %s ..." % (i + 1, head))


def step2_search_whole(client, ollama_ef, question):
    print()
    print("[2] 청킹 없이(통짜) 검색")

    collection = reset_collection(client, ollama_ef, "whole")
    collection.add(ids=["whole-1"], documents=[DOC])
    print("  문서 '전체'를 단 1개의 document로 저장했습니다. (문서 수: %d)" % collection.count())

    result = collection.query(query_texts=[question], n_results=1)
    distance = result["distances"][0][0]
    found = result["documents"][0][0]

    print("  질문: %s" % question)
    print("  1위 거리: %.4f" % distance)
    print("  돌려받은 문서(앞부분): %s ..." % found[:40].replace("\n", " "))
    print("  -> 돌려받은 건 '환불 문단'이 아니라 문서 전체입니다. 통짜라 그것 말고 줄 게 없습니다.")
    print("  문제점: 5개 주제(배송/교환/멤버십/영업시간/환불)가 한 벡터에 섞여 평균값처럼 흐려집니다.")
    print("          그래서 '환불' 문단만 따로 담았을 때보다 거리가 뜨고(뒤 [3]과 비교),")
    print("          원하는 '환불' 부분만 콕 집어 돌려주지도 못합니다.")

    return distance


def step3_search_chunks(client, ollama_ef, chunks, question, whole_distance):
    print()
    print("[3] 청킹 후 검색")

    collection = reset_collection(client, ollama_ef, "chunks")

    ids = []
    for i in range(len(chunks)):
        ids.append("chunk-%d" % (i + 1))
    collection.add(ids=ids, documents=chunks)
    print("  청크 %d개를 각각 따로 저장했습니다. (add 시 Chroma가 자동 임베딩)" % collection.count())

    result = collection.query(query_texts=[question], n_results=2)
    result_ids = result["ids"][0]
    result_docs = result["documents"][0]
    result_distances = result["distances"][0]

    print("  질문: %s" % question)
    for rank in range(len(result_ids)):
        print("    %d위 | 거리 %.4f | %s" % (rank + 1, result_distances[rank], result_ids[rank]))
        print("         %s" % result_docs[rank])

    top_distance = result_distances[0]
    print("  통짜 거리 %.4f  vs  청킹 1위 거리 %.4f" % (whole_distance, top_distance))
    print("  -> 청킹하면 '환불' 문단만 정확히 1위로 나오고, 거리도 더 낮아(더 가까워) 판단이 뚜렷해집니다.")

    return collection


def search(collection, query, k):
    print("  질문: %s" % query)

    result = collection.query(query_texts=[query], n_results=k)
    result_docs = result["documents"][0]
    result_distances = result["distances"][0]

    for rank in range(len(result_docs)):
        print("    %d위 | 거리 %.4f" % (rank + 1, result_distances[rank]))
        print("         %s" % result_docs[rank])

    top_distance = result_distances[0]
    print("  1위 거리 %.4f  (기준 threshold %.2f)" % (top_distance, DISTANCE_THRESHOLD))
    if top_distance < DISTANCE_THRESHOLD:
        print("  판정: 관련 문서를 찾았습니다.")
        print("        답이 될 문단 -> %s" % result_docs[0])
    else:
        print("  판정: 관련 문서를 찾지 못했습니다. (문서에 없는 내용으로 보입니다)")


def step4_mini_search(collection):
    print()
    print("[4] 미니 검색기 + 거리 해석(진단)")
    print("  규칙: 1위 거리가 %.2f 보다 작으면 '관련 있음', 크면 '관련 문서 없음'." % DISTANCE_THRESHOLD)

    print()
    print("  (가) 관련 질문:")
    search(collection, "교환하려면 어떻게 하나요?", 2)

    print()
    print("  (나) 무관 질문:")
    search(collection, "로켓 발사 절차를 알려줘", 2)

    print()
    print("  교훈: 거리에는 절대적인 '좋다/나쁘다' 기준선이 없습니다.")
    print("        관련 문서는 거리가 낮고 무관한 질문은 높다는 '경향'만 있을 뿐,")
    print("        그 경계선(threshold)은 모델과 데이터로 직접 돌려보며 경험적으로 정합니다.")


def main():
    if not check_ollama():
        sys.exit(1)

    client = chromadb.PersistentClient(path="./chroma_db")
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url=OLLAMA_URL,
        model_name=EMBED_MODEL,
    )

    question = "환불은 며칠 걸리나요?"

    chunks = split_into_paragraphs(DOC)

    step1_show_chunking(chunks)
    whole_distance = step2_search_whole(client, ollama_ef, question)
    chunks_collection = step3_search_chunks(client, ollama_ef, chunks, question, whole_distance)
    step4_mini_search(chunks_collection)


if __name__ == "__main__":
    main()