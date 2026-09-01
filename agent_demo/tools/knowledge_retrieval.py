from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class KnowledgeRetrievalTool:
    """Small dependency-free retriever for Markdown fault knowledge.

    This intentionally uses lexical scoring so the demo remains reproducible
    without an embedding model or external API. It can later be replaced by
    FAISS/sentence-transformers without changing the Agent interface.
    """

    def __init__(self, knowledge_dir: str | Path | None = None):
        if knowledge_dir is None:
            knowledge_dir = Path(__file__).resolve().parents[1] / "knowledge"
        self.knowledge_dir = Path(knowledge_dir)
        self.documents = self._load_documents()

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]+", text)
            if len(token.strip()) > 1
        }

    def _load_documents(self) -> list[dict[str, str]]:
        docs: list[dict[str, str]] = []
        if not self.knowledge_dir.exists():
            return docs

        for path in sorted(self.knowledge_dir.glob("*.md")):
            if path.name.lower() == "readme.md":
                continue
            text = path.read_text(encoding="utf-8")
            docs.append({"source": path.name, "content": text})
        return docs

    def search(self, query: str, top_k: int = 2) -> list[dict[str, Any]]:
        query_tokens = self._tokens(query)
        scored: list[dict[str, Any]] = []

        for doc in self.documents:
            doc_tokens = self._tokens(doc["content"])
            overlap = query_tokens & doc_tokens
            score = len(overlap) / max(len(query_tokens), 1)
            scored.append(
                {
                    "source": doc["source"],
                    "score": round(float(score), 4),
                    "content": doc["content"],
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[: max(int(top_k), 1)]
