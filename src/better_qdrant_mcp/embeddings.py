from __future__ import annotations

from typing import Iterable, List
import re

import jieba
from fastembed import SparseTextEmbedding
from openai import OpenAI

from .config import get_settings


class Embeddings:
    _client: OpenAI | None = None

    @classmethod
    def client(cls) -> OpenAI:
        if cls._client is None:
            s = get_settings()
            if not s.openai_api_key:
                raise RuntimeError(
                    "OpenAI API key is required (OPENAPI_API_KEY or OPENAI_API_KEY)"
                )
            cls._client = OpenAI(
                api_key=s.openai_api_key,
                base_url=s.openai_base_url,
                timeout=s.openai_timeout,
                max_retries=2,
            )
        return cls._client

    @classmethod
    def embed_one(cls, text: str) -> List[float]:
        s = get_settings()
        resp = cls.client().embeddings.create(
            model=s.openai_embedding_model,
            input=text,
        )
        return list(resp.data[0].embedding)

    @classmethod
    def embed_many(cls, texts: Iterable[str]) -> List[List[float]]:
        s = get_settings()
        # OpenAI API supports batching, but to keep behavior simple, call once with list
        resp = cls.client().embeddings.create(
            model=s.openai_embedding_model,
            input=list(texts),
        )
        return [list(item.embedding) for item in resp.data]


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _preprocess_text(text: str) -> str:
    text = text.strip()
    if _CJK_RE.search(text):
        tokens = jieba.lcut(text)
        return " ".join(t for t in tokens if t.strip())
    return " ".join(text.lower().split())


class SparseEmbeddings:
    _model: SparseTextEmbedding | None = None

    @classmethod
    def model(cls) -> SparseTextEmbedding:
        if cls._model is None:
            cls._model = SparseTextEmbedding(model_name="Qdrant/bm25")
        return cls._model

    @classmethod
    def embed_one(cls, text: str) -> tuple[List[int], List[float]]:
        processed = _preprocess_text(text)
        for emb in cls.model().embed([processed]):
            return list(emb.indices), list(emb.values)
        return [], []

    @classmethod
    def embed_many(cls, texts: Iterable[str]) -> List[tuple[List[int], List[float]]]:
        processed = [_preprocess_text(t) for t in texts]
        out: List[tuple[List[int], List[float]]] = []
        for emb in cls.model().embed(processed):
            out.append((list(emb.indices), list(emb.values)))
        return out
