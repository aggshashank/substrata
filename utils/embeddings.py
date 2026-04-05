"""Substrata vector store — embedding generation and ChromaDB storage for RAG retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
import litellm
from loguru import logger

from config import get_settings


PrimitiveMetadataValue = str | int | float | bool


class WikiVectorStore:
    """Manage Substrata chunk embeddings in a local ChromaDB collection."""

    collection_name = "wiki_chunks"

    def __init__(self, chroma_dir: Path | None = None) -> None:
        self.settings = get_settings()
        self.chroma_dir = chroma_dir or self.settings.CHROMA_DIR
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.embed_model = self.settings.EMBED_MODEL
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _generate_embedding(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts via Ollama through LiteLLM."""
        if not texts:
            return []

        try:
            response = litellm.embedding(
                model=self.embed_model,
                input=texts,
                api_base=self.settings.OLLAMA_BASE_URL,
            )
            response_data = response.get("data", []) if isinstance(response, dict) else getattr(response, "data", [])
            embeddings: list[list[float]] = []

            for item in response_data:
                embedding = item.get("embedding") if isinstance(item, dict) else getattr(item, "embedding", None)
                if embedding is None:
                    logger.warning("Embedding response missing vector payload for model {}", self.embed_model)
                    return []
                embeddings.append([float(value) for value in embedding])

            if len(embeddings) != len(texts):
                logger.warning(
                    "Embedding count mismatch for model {}: expected={} actual={}",
                    self.embed_model,
                    len(texts),
                    len(embeddings),
                )
                return []

            return embeddings
        except Exception as exc:
            logger.warning("Embedding generation failed for {} text(s): {}", len(texts), exc)
            return []

    def embed_and_store(self, doc_id: str, chunks: list[str], metadata: list[dict[str, Any]]) -> int:
        """Embed chunks and upsert them into the ChromaDB collection."""
        if not chunks:
            return 0
        if len(chunks) != len(metadata):
            logger.warning(
                "Chunk and metadata length mismatch for doc {}: chunks={} metadata={}",
                doc_id,
                len(chunks),
                len(metadata),
            )
            return 0

        embeddings = self._generate_embedding(chunks)
        if len(embeddings) != len(chunks):
            return 0

        ids = [f"{doc_id}_chunk_{index}" for index in range(len(chunks))]
        flattened_metadata = [
            self._flatten_metadata({"doc_id": doc_id, **chunk_metadata})
            for chunk_metadata in metadata
        ]

        try:
            self.collection.upsert(
                ids=ids,
                documents=chunks,
                metadatas=flattened_metadata,
                embeddings=embeddings,
            )
            logger.info("Stored {} chunk embeddings for doc {}", len(chunks), doc_id)
            return len(chunks)
        except Exception as exc:
            logger.warning("Failed to store embeddings for doc {}: {}", doc_id, exc)
            return 0

    def query_similar(self, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        """Return similar chunks for a text query."""
        if not query.strip():
            return []

        try:
            if self.collection.count() == 0:
                return []
        except Exception as exc:
            logger.warning("Failed to inspect collection {}: {}", self.collection_name, exc)
            return []

        query_embeddings = self._generate_embedding([query])
        if not query_embeddings:
            return []

        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
            documents = results.get("documents", [[]])
            metadatas = results.get("metadatas", [[]])
            distances = results.get("distances", [[]])

            matched_documents = documents[0] if documents else []
            matched_metadata = metadatas[0] if metadatas else []
            matched_distances = distances[0] if distances else []

            matches: list[dict[str, Any]] = []
            for index, document in enumerate(matched_documents):
                metadata = matched_metadata[index] if index < len(matched_metadata) else {}
                distance = matched_distances[index] if index < len(matched_distances) else None
                matches.append(
                    {
                        "text": document,
                        "metadata": metadata or {},
                        "distance": distance,
                    }
                )
            return matches
        except Exception as exc:
            logger.warning("Similarity query failed: {}", exc)
            return []

    def delete_doc(self, doc_id: str) -> int:
        """Delete all stored chunks for a document id and return the count removed."""
        if not doc_id:
            return 0

        try:
            existing = self.collection.get(where={"doc_id": doc_id})
            ids = existing.get("ids", []) if isinstance(existing, dict) else []
            if not ids:
                return 0
            self.collection.delete(ids=ids)
            logger.info("Deleted {} chunk embeddings for doc {}", len(ids), doc_id)
            return len(ids)
        except Exception as exc:
            logger.warning("Failed to delete embeddings for doc {}: {}", doc_id, exc)
            return 0

    def get_stats(self) -> dict[str, Any]:
        """Return basic collection statistics."""
        try:
            total_chunks = self.collection.count()
        except Exception as exc:
            logger.warning("Failed to count collection {}: {}", self.collection_name, exc)
            total_chunks = 0

        return {
            "total_chunks": total_chunks,
            "collection_name": self.collection_name,
        }

    def _flatten_metadata(self, metadata: dict[str, Any]) -> dict[str, PrimitiveMetadataValue]:
        """Convert metadata values into ChromaDB-compatible primitive types."""
        flattened: dict[str, PrimitiveMetadataValue] = {}
        for key, value in metadata.items():
            normalized = self._normalize_metadata_value(value)
            if normalized is None:
                continue
            flattened[key] = normalized
        return flattened

    def _normalize_metadata_value(self, value: Any) -> PrimitiveMetadataValue | None:
        """Normalize arbitrary values into primitive ChromaDB metadata values."""
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (str, int, float)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            parts = [
                f"{nested_key}={self._stringify_metadata_value(nested_value)}"
                for nested_key, nested_value in value.items()
                if nested_value is not None
            ]
            return ", ".join(part for part in parts if part)
        if isinstance(value, (list, tuple, set)):
            parts = [self._stringify_metadata_value(item) for item in value if item is not None]
            return ", ".join(part for part in parts if part)
        return str(value)

    def _stringify_metadata_value(self, value: Any) -> str:
        """Stringify nested metadata fragments for flattened storage."""
        normalized = self._normalize_metadata_value(value)
        if normalized is None:
            return ""
        return str(normalized)
