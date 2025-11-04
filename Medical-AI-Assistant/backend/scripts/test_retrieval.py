#!/usr/bin/env python3
"""
Test RAG Retrieval

Test the RAG retrieval system with sample queries.

Usage:
    python scripts/test_retrieval.py
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
from app.rag.retriever import Retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test queries
TEST_QUERIES = [
    "What are red flag symptoms for chest pain?",
    "How should I triage a patient with severe headache?",
    "What questions should I ask about abdominal pain?",
    "When should I escalate to emergency care?",
    "What are the signs of a stroke?",
]


async def test_retrieval():
    """Test retrieval with sample queries"""
    logger.info("Initializing RAG system")
    
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
    
    retriever = Retriever(
        vector_store=vector_store,
        embeddings_service=embeddings_service,
        config=rag_config
    )
    
    # Test each query
    logger.info(f"\nTesting {len(TEST_QUERIES)} queries\n")
    logger.info("=" * 80)
    
    for i, query in enumerate(TEST_QUERIES, 1):
        logger.info(f"\nQuery {i}: {query}")
        logger.info("-" * 80)
        
        results = await retriever.retrieve(query, top_k=3)
        
        if results:
            logger.info(f"Found {len(results)} relevant documents:")
            for j, result in enumerate(results, 1):
                logger.info(f"\n  Result {j}:")
                logger.info(f"    Source: {result.get('metadata', {}).get('source', 'unknown')}")
                logger.info(f"    Score: {result.get('score', 0):.3f}")
                logger.info(f"    Snippet: {result.get('text', '')[:200]}...")
        else:
            logger.info("  No results found")
        
        logger.info("")
    
    logger.info("=" * 80)
    logger.info("\nRetrieval test complete!")


if __name__ == "__main__":
    logger.info("NOTE: This is a placeholder test script.")
    logger.info("Full implementation requires vector store initialization.")
    logger.info("Will be fully functional after model integration.\n")
    
    asyncio.run(test_retrieval())
