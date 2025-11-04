#!/usr/bin/env python3
"""
Ingest Medical Guidelines into Vector Store

This script ingests markdown guidelines from the data/guidelines directory
into the Chroma vector store for RAG retrieval.

Usage:
    python scripts/ingest_guidelines.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.rag.config import RAGConfig
from app.rag.embeddings import EmbeddingsService
from app.rag.vector_store import VectorStoreManager
from app.rag.document_processor import DocumentProcessor
from app.rag.ingestion import IngestionPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main ingestion function"""
    logger.info("Starting medical guidelines ingestion")
    
    # Initialize components
    rag_config = RAGConfig.from_settings(settings)
    
    embeddings_service = EmbeddingsService(
        model_name=rag_config.embedding_model,
        device=rag_config.embedding_device
    )
    
    vector_store = VectorStoreManager(
        persist_directory=rag_config.vector_store_path,
        embeddings_service=embeddings_service
    )
    
    document_processor = DocumentProcessor(
        chunk_size=rag_config.chunk_size,
        chunk_overlap=rag_config.chunk_overlap
    )
    
    # Create ingestion pipeline
    pipeline = IngestionPipeline(
        document_processor=document_processor,
        embeddings_service=embeddings_service,
        vector_store=vector_store
    )
    
    # Ingest guidelines directory
    guidelines_dir = Path("data/guidelines")
    
    if not guidelines_dir.exists():
        logger.error(f"Guidelines directory not found: {guidelines_dir}")
        return
    
    logger.info(f"Ingesting guidelines from: {guidelines_dir}")
    await pipeline.ingest_directory(guidelines_dir)
    
    logger.info("Ingestion complete!")
    logger.info(f"Vector store saved to: {rag_config.vector_store_path}")


if __name__ == "__main__":
    asyncio.run(main())
