import math


def fixed_char(text, size):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start = start + size
    return chunks


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


def recursive_split_overlap(text, max_size, overlap):
    # overlap = 새 청크를 시작할 때 물려줄 '이전 청크의 끝 문장' 개수
    sentences = split_sentences(text)
    chunks = []
    current = []
    for s in sentences:
        if current and len(" ".join(current + [s])) > max_size:
            chunks.append(" ".join(current))
            if overlap > 0:
                carry = current[len(current) - overlap:]
            else:
                carry = []
            current = carry + [s]
        else:
            current.append(s)
    if current:
        chunks.append(" ".join(current))
    return chunks


def cosine(a, b):
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(len(a)):
        dot = dot + a[i] * b[i]
        na = na + a[i] * a[i]
        nb = nb + b[i] * b[i]
    return dot / (math.sqrt(na) * math.sqrt(nb))


def semantic_split(text, embed_fn, drop):
    sentences = split_sentences(text)
    embs = embed_fn(sentences)
    chunks = []
    current = sentences[0]
    for i in range(1, len(sentences)):
        if cosine(embs[i - 1], embs[i]) < drop:
            chunks.append(current)
            current = sentences[i]
        else:
            current = current + " " + sentences[i]
    chunks.append(current)
    return chunks


def parent_child_split(text, parent_max):
    parents = recursive_split(text, parent_max)
    children = []
    for pi in range(len(parents)):
        for sent in split_sentences(parents[pi]):
            children.append((sent, pi))
    return parents, children
