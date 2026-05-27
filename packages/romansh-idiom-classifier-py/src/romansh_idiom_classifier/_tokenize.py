import re

TOKEN_RE = re.compile(r"\w{2,}", re.UNICODE)


def char_wb_ngrams(text: str, min_n: int, max_n: int) -> list[str]:
    ngrams: list[str] = []
    for token in TOKEN_RE.findall(text.lower()):
        padded = " " + token + " "
        for n in range(min_n, max_n + 1):
            for i in range(len(padded) - n + 1):
                ngrams.append(padded[i : i + n])
    return ngrams


def word_ngrams(text: str, min_n: int, max_n: int) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    ngrams: list[str] = []
    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            ngrams.append(" ".join(tokens[i : i + n]))
    return ngrams
