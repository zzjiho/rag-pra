import numpy as np
import ollama

MODEL = "bge-m3"


def embed(text):
    response = ollama.embeddings(model=MODEL, prompt=text)
    return np.array(response["embedding"])


def cosine_similarity(a, b):
    dot = np.dot(a, b)
    return dot / (np.linalg.norm(a) * np.linalg.norm(b))


def l2_distance(a, b):
    diff = a - b
    return np.linalg.norm(diff)


def step1_embedding():
    print("[1] 문장 -> 벡터 변환")
    sentence = "강아지가 공원에서 신나게 뛰어놀고 있다."
    vec = embed(sentence)

    preview = []
    for value in vec[:5]:
        preview.append(round(float(value), 4))
    norm = round(float(np.linalg.norm(vec)), 4)

    print("  원본 문장     :", sentence)
    print("  벡터 차원(len):", len(vec))
    print("  앞 5개 숫자   :", preview)
    print("  벡터 크기(norm):", norm)
    print("  -> 문장의 '의미'가", len(vec), "개의 숫자(좌표)로 바뀌었다.")
    print("  -> norm이 1.0에 가까우면 정규화된 벡터, 아니면 원시 벡터다.")
    print()


def step2_similarity():
    print("[2] 유사도 3종 비교")
    base = "강아지가 공원에서 뛰어논다."
    similar = "개가 마당에서 신나게 달리고 있다."
    unrelated = "어제 주식 시장이 크게 하락했다."

    base_vec = embed(base)
    similar_vec = embed(similar)
    unrelated_vec = embed(unrelated)

    print("  기준 문장:", base)
    print()
    compare_pair("뜻이 비슷한 문장", similar, base_vec, similar_vec)
    compare_pair("상관없는 문장", unrelated, base_vec, unrelated_vec)
    print("  -> 뜻이 비슷할수록 코사인은 커지고(1에 가깝고), L2 거리는 작아진다.")
    print()


def compare_pair(label, sentence, base_vec, other_vec):
    cosine = cosine_similarity(base_vec, other_vec)
    l2 = l2_distance(base_vec, other_vec)
    dot = np.dot(base_vec, other_vec)

    print(f"  [{label}] {sentence}")
    print(f"    코사인 유사도 : {cosine:.4f}")
    print(f"    L2 거리       : {l2:.4f}")
    print(f"    내적(dot)     : {dot:.4f}")
    print()


def step3_search():
    print("[3] 의미 검색 데모 (벡터 DB 없이 손으로 전수 비교)")
    sentences = [
        "결제 취소 후 3~5영업일 내에 환불 금액이 입금됩니다.",
        "전원 버튼을 길게 눌러도 노트북이 켜지지 않을 때의 조치입니다.",
        "이번 분기 매출이 전년 대비 20퍼센트 증가했다.",
        "강아지 산책은 하루 두 번, 아침과 저녁에 시켜주세요.",
        "김치찌개는 돼지고기와 잘 익은 김치로 끓여야 맛있다.",
        "회원 등급이 올라가면 무료 배송 혜택이 제공됩니다.",
    ]
    query = "환불은 언제 돼요?"

    query_vec = embed(query)

    results = []
    for sentence in sentences:
        vec = embed(sentence)
        result = {
            "sentence": sentence,
            "cosine": cosine_similarity(query_vec, vec),
            "l2": l2_distance(query_vec, vec),
        }
        results.append(result)

    results.sort(key=lambda item: item["cosine"], reverse=True)

    print("  질문:", query)
    print("  (질문과 각 문장의 코사인 유사도를 재서 높은 순으로 정렬)")
    print()
    for rank, result in enumerate(results, start=1):
        print(f"  {rank}위  코사인 {result['cosine']:.4f}  |  L2 {result['l2']:.4f}  |  {result['sentence']}")
    print()

    return query, results[0]["sentence"]


def step4_keyword(query, top_sentence):
    print("[4] 키워드 검색이었다면?")
    print("  질문     :", query)
    print("  1위 문장 :", top_sentence)

    query_words = query.split()
    overlap = []
    for word in query_words:
        if word in top_sentence:
            overlap.append(word)

    print("  질문을 공백으로 나눈 단어         :", query_words)
    print("  1위 문장에 그대로 들어있는 단어   :", overlap)
    print()
    if len(overlap) == 0:
        print("  -> 겹치는 단어가 하나도 없다.")
    else:
        print("  -> 겹치는 단어가", len(overlap), "개뿐이다.")
    print("  -> 키워드 검색이었다면 이 문장을 못 찾았을 것이다.")
    print("  -> 단어가 안 겹쳐도 '뜻'이 가까워서 의미 검색은 찾아냈다.")
    print()


def main():
    try:
        embed("연결 확인용 문장")
    except Exception as error:
        print("Ollama에서 임베딩을 가져오지 못했습니다.")
        print("아래를 확인하세요.")
        print("  1) Ollama 서버가 실행 중인가?  (brew services start ollama 또는 Ollama 앱 실행)")
        print("  2) 모델이 설치돼 있는가?       ollama pull bge-m3")
        print("원본 오류:", error)
        return

    step1_embedding()
    step2_similarity()
    query, top_sentence = step3_search()
    step4_keyword(query, top_sentence)


if __name__ == "__main__":
    main()