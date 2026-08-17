import json
import sys
import urllib.error
import urllib.request

import chromadb
from chromadb.utils import embedding_functions


OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "bge-m3"


def check_ollama():
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=5) as res:
            data = json.loads(res.read())
    except urllib.error.URLError:
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


def step1_prepare(client, ollama_ef):
    print()
    print("[1] 준비 & 영속성 확인")

    collection = client.get_or_create_collection(
        name="faq",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )

    print("  지난 실행에서 남아있던 문서 수: %d 개" % collection.count())
    print("  -> 0보다 크면, 저번에 저장한 데이터가 ./chroma_db 디스크에 그대로 남아있었다는 뜻(영속성).")

    print("  이번 데모를 깔끔히 보기 위해 컬렉션을 초기화합니다.")
    client.delete_collection(name="faq")
    collection = client.get_or_create_collection(
        name="faq",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )
    print("  초기화 후 문서 수: %d 개" % collection.count())

    return collection


def step2_add(collection):
    print()
    print("[2] 생성 (add)")

    ids = ["faq-1", "faq-2", "faq-3", "faq-4", "faq-5", "faq-6"]
    documents = [
        "환불은 상품 수령 후 7일 이내에 신청하실 수 있으며, 승인되면 3영업일 내에 처리됩니다.",
        "주문하신 상품은 결제 완료 후 보통 2~3일 이내에 배송됩니다.",
        "비밀번호를 잊으셨다면 로그인 화면의 '비밀번호 찾기'를 눌러 이메일로 재설정할 수 있습니다.",
        "노트북 전원이 켜지지 않으면 충전 어댑터를 연결하고 전원 버튼을 10초간 길게 눌러 주세요.",
        "고객센터 영업시간은 평일 오전 9시부터 오후 6시까지이며, 주말과 공휴일은 휴무입니다.",
        "멤버십에 가입하시면 무료 배송과 포인트 적립 등 다양한 혜택을 받으실 수 있습니다.",
    ]
    metadatas = [
        {"topic": "환불"},
        {"topic": "배송"},
        {"topic": "계정"},
        {"topic": "기기"},
        {"topic": "영업시간"},
        {"topic": "멤버십"},
    ]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    print("  FAQ 문서 %d개를 저장했습니다." % len(ids))
    print("  Phase 0에서는 embed()를 손으로 불러 벡터를 만들었지만,")
    print("  이제 add만 하면 ChromaDB가 각 문서를 bge-m3로 자동 임베딩해서 저장합니다.")
    print("  현재 문서 수: %d 개" % collection.count())


def step3_get(collection):
    print()
    print("[3] 조회 (get)")

    result = collection.get(ids=["faq-2"])
    print("  id 'faq-2' 로 정확히 꺼냈습니다.")
    print("    문서    : %s" % result["documents"][0])
    print("    메타데이터: %s" % result["metadatas"][0])
    print("  get은 '정확한 id로 꺼내기'입니다. 의미가 비슷한 걸 찾는 검색과는 다릅니다.")


def step4_query(collection):
    print()
    print("[4] 의미 검색 (query)")

    question = "환불은 언제 돼요?"
    result = collection.query(query_texts=[question], n_results=3)

    ids = result["ids"][0]
    documents = result["documents"][0]
    distances = result["distances"][0]
    metadatas = result["metadatas"][0]

    print("  질문: %s" % question)
    print("  질문과 의미가 가까운 순서로 3개를 찾았습니다 (거리는 낮을수록 가깝다).")
    for rank in range(len(ids)):
        print("    %d위 | 거리 %.4f | topic=%s" % (rank + 1, distances[rank], metadatas[rank]["topic"]))
        print("         %s" % documents[rank])
    print("  1위가 환불 문서면 성공. Phase 0에서 손으로 전수 비교하던 걸 Chroma가 대신 했습니다.")


def step5_update(collection):
    print()
    print("[5] 수정 (update)")

    new_text = "고객센터 영업시간은 평일 오전 9시부터 오후 6시까지이며, 토요일은 오전 9시부터 오후 1시까지 운영합니다."
    collection.update(ids=["faq-5"], documents=[new_text])
    print("  id 'faq-5'(영업시간) 문서 내용을 수정했습니다.")

    result = collection.get(ids=["faq-5"])
    print("  수정 후 다시 조회:")
    print("    %s" % result["documents"][0])


def step6_delete(collection):
    print()
    print("[6] 삭제 (delete)")

    print("  삭제 전 문서 수: %d 개" % collection.count())
    collection.delete(ids=["faq-4"])
    print("  id 'faq-4'(노트북 전원) 문서를 삭제했습니다.")
    print("  삭제 후 문서 수: %d 개" % collection.count())


def step7_outro():
    print()
    print("[7] 마무리 안내")
    print("  방금 다룬 데이터는 ./chroma_db 폴더에 저장되어 있습니다.")
    print("  이 파일을 다시 실행하면 [1]에서 '남아있던 문서 수'로 그 데이터가 보입니다.")


def main():
    if not check_ollama():
        sys.exit(1)

    client = chromadb.PersistentClient(path="./chroma_db")
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url=OLLAMA_URL,
        model_name=EMBED_MODEL,
    )

    collection = step1_prepare(client, ollama_ef)
    step2_add(collection)
    step3_get(collection)
    step4_query(collection)
    step5_update(collection)
    step6_delete(collection)
    step7_outro()


if __name__ == "__main__":
    main()