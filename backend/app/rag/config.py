"""RAG Configuration"""
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    
    # Embedding Model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_dimension: int = 384
    
    # Vector Store
    vector_store_type: str = "chroma"
    vector_store_path: str = "data/vector_store"
    collection_name: str = "medical_guidelines"
    
    # Retrieval
    top_k: int = 5
    similarity_threshold: float = 0.7
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Document Processing
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    @classmethod
    def from_settings(cls, settings):
        """Create from application settings"""
        return cls(
            embedding_model=settings.EMBEDDING_MODEL,
            embedding_device=settings.EMBEDDING_DEVICE,
            vector_store_path=settings.VECTOR_STORE_PATH,
            top_k=settings.RAG_TOP_K,
            similarity_threshold=settings.RAG_SIMILARITY_THRESHOLD,
            use_reranker=settings.RAG_USE_RERANKER,
            reranker_model=settings.RAG_RERANKER_MODEL,
        )
