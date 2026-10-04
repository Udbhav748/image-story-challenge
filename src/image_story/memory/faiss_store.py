"""FAISS vector store for semantic memory and retrieval."""
import os
import pickle
import uuid
from typing import Any
import numpy as np
import faiss

from ..domain.schemas import EvidenceRecord, RetrievedEvidence
from ..domain.exceptions import FAISSIndexError, RetrievalError
from .embeddings import EmbeddingModelInterface


class FAISSVectorStore:
    """FAISS-based vector store for evidence retrieval."""
    
    def __init__(
        self,
        embedding_model: EmbeddingModelInterface,
        index_type: str = "flat_ip",
        device: str = "cpu",
    ):
        self._embedding_model = embedding_model
        self._index_type = index_type
        self._device = device
        self._index: faiss.Index | None = None
        self._evidence_map: dict[int, EvidenceRecord] = {}
        self._id_to_index: dict[str, int] = {}
        self._next_index = 0
        self._dimension = embedding_model.embedding_dim
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def count(self) -> int:
        return self._next_index
    
    def _create_index(self) -> faiss.Index:
        if self._index_type == "flat_ip":
            index = faiss.IndexFlatIP(self._dimension)
        elif self._index_type == "flat_l2":
            index = faiss.IndexFlatL2(self._dimension)
        elif self._index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self._dimension, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 128
        else:
            raise FAISSIndexError(f"Unknown index type: {self._index_type}")
        
        if self._device == "cuda" and hasattr(faiss, "StandardGpuResources"):
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception:
                pass
        
        return index
    
    def initialize(self) -> None:
        """Initialize the FAISS index."""
        if self._index is None:
            self._index = self._create_index()
    
    def add_evidence(self, evidence: EvidenceRecord, text: str) -> int:
        """Add evidence to the store with its embedding text."""
        self.initialize()
        
        embedding = self._embedding_model.encode_single(text)
        embedding = embedding.reshape(1, -1)
        
        index_id = self._next_index
        self._index.add(embedding)
        self._evidence_map[index_id] = evidence
        self._id_to_index[evidence.id] = index_id
        self._next_index += 1
        
        return index_id
    
    def add_batch(self, evidence_list: list[EvidenceRecord], texts: list[str]) -> list[int]:
        """Add multiple evidence records at once."""
        self.initialize()
        
        if not evidence_list:
            return []
        
        embeddings = self._embedding_model.encode(texts)
        index_ids = list(range(self._next_index, self._next_index + len(evidence_list)))
        
        self._index.add(embeddings)
        for i, (evidence, idx) in enumerate(zip(evidence_list, index_ids)):
            self._evidence_map[idx] = evidence
            self._id_to_index[evidence.id] = idx
        
        self._next_index += len(evidence_list)
        return index_ids
    
    def search(
        self,
        query_text: str,
        top_k: int = 10,
        filter_fn: callable = None,
    ) -> list[RetrievedEvidence]:
        """Search for similar evidence."""
        if self._index is None or self._next_index == 0:
            return []
        
        query_embedding = self._embedding_model.encode_single(query_text)
        query_embedding = query_embedding.reshape(1, -1)
        
        actual_k = min(top_k * 2, self._next_index)
        scores, indices = self._index.search(query_embedding, actual_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= self._next_index:
                continue
            
            evidence = self._evidence_map.get(idx)
            if evidence is None:
                continue
            
            if filter_fn and not filter_fn(evidence):
                continue
            
            results.append(RetrievedEvidence(
                record=evidence,
                semantic_similarity=float(score),
                rank_score=0.0,
                rank_factors={"semantic_similarity": float(score)},
            ))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_fn: callable = None,
    ) -> list[RetrievedEvidence]:
        """Search using a pre-computed embedding."""
        if self._index is None or self._next_index == 0:
            return []
        
        query_embedding = query_embedding.reshape(1, -1)
        actual_k = min(top_k * 2, self._next_index)
        scores, indices = self._index.search(query_embedding, actual_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= self._next_index:
                continue
            
            evidence = self._evidence_map.get(idx)
            if evidence is None:
                continue
            
            if filter_fn and not filter_fn(evidence):
                continue
            
            results.append(RetrievedEvidence(
                record=evidence,
                semantic_similarity=float(score),
                rank_score=0.0,
                rank_factors={"semantic_similarity": float(score)},
            ))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def get_evidence(self, index_id: int) -> EvidenceRecord | None:
        """Get evidence by internal index ID."""
        return self._evidence_map.get(index_id)
    
    def get_evidence_by_id(self, evidence_id: str) -> EvidenceRecord | None:
        """Get evidence by evidence ID."""
        idx = self._id_to_index.get(evidence_id)
        if idx is not None:
            return self._evidence_map.get(idx)
        return None
    
    def remove_evidence(self, evidence_id: str) -> bool:
        """Remove evidence from the store (marks as deleted, doesn't shrink index)."""
        idx = self._id_to_index.pop(evidence_id, None)
        if idx is not None:
            self._evidence_map.pop(idx, None)
            return True
        return False
    
    def save(self, path: str) -> None:
        """Save the index and metadata to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        index_path = f"{path}.index"
        meta_path = f"{path}.meta"
        
        if self._index is not None:
            faiss.write_index(self._index, index_path)
        
        with open(meta_path, "wb") as f:
            pickle.dump({
                "evidence_map": self._evidence_map,
                "id_to_index": self._id_to_index,
                "next_index": self._next_index,
                "dimension": self._dimension,
                "index_type": self._index_type,
            }, f)
    
    def load(self, path: str) -> None:
        """Load the index and metadata from disk."""
        index_path = f"{path}.index"
        meta_path = f"{path}.meta"
        
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FAISSIndexError(f"Index files not found at {path}")
        
        self._index = faiss.read_index(index_path)
        
        with open(meta_path, "rb") as f:
            data = pickle.load(f)
            self._evidence_map = data["evidence_map"]
            self._id_to_index = data["id_to_index"]
            self._next_index = data["next_index"]
            self._dimension = data["dimension"]
            self._index_type = data["index_type"]
    
    def clear(self) -> None:
        """Clear the store."""
        self._index = self._create_index()
        self._evidence_map.clear()
        self._id_to_index.clear()
        self._next_index = 0
    
    def get_stats(self) -> dict[str, Any]:
        return {
            "count": self._next_index,
            "dimension": self._dimension,
            "index_type": self._index_type,
            "device": self._device,
        }