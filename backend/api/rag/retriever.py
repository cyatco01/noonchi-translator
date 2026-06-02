"""
Sociolinguistic RAG retriever backed by ChromaDB.

Initialized once at startup (inside ClaudeTranslationAgent.__init__).
The in-memory collection is populated from the static knowledge base;
queries are filtered by formality token so only relevant notes are returned.
"""

import logging

import chromadb
import chromadb.errors

from .knowledge_base import ENTRIES

logger = logging.getLogger(__name__)


class SociolinguisticRetriever:
    """
    Vector-similarity retriever over the sociolinguistic knowledge base.

    Filters by formality token at query time so retrieved notes always
    match the requested speech level.
    """

    def __init__(self):
        client = chromadb.Client()
        self._collection = client.create_collection("sociolinguistic_notes")
        self._populate()
        self._verify_reachable()
        logger.info(f"RAG retriever initialized with {len(ENTRIES)} knowledge base entries")

    def _verify_reachable(self) -> None:
        """Smoke-test the collection at startup to catch misconfiguration early."""
        try:
            self._collection.query(
                query_texts=["test"],
                n_results=1,
                where={"applies_formal": True},
            )
        except chromadb.errors.ChromaError as e:
            raise RuntimeError(
                f"RAG collection failed startup verification: {e}"
            ) from e

    def _populate(self):
        self._collection.add(
            ids=[e["id"] for e in ENTRIES],
            documents=[e["text"] for e in ENTRIES],
            metadatas=[
                {
                    "applies_formal": "formal" in e["applies_to"],
                    "applies_polite": "polite" in e["applies_to"],
                    "applies_casual": "casual" in e["applies_to"],
                }
                for e in ENTRIES
            ],
        )

    def retrieve(self, query: str, formality_token: str, n_results: int = 3) -> list[str]:
        """
        Return the top-n relevant notes for a given query and formality level.

        Args:
            query: English source text (used as the embedding query)
            formality_token: "formal", "polite", or "casual"
            n_results: how many notes to return

        Returns:
            List of note strings, ordered by relevance.
        """
        where = {f"applies_{formality_token}": True}
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
            )
            return results["documents"][0] if results["documents"] else []
        except chromadb.errors.ChromaError:
            logger.warning("RAG retrieval failed; continuing without augmentation", exc_info=True)
            return []
