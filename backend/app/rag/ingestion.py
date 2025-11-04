"""Ingestion Pipeline"""
import logging
from typing import List
from pathlib import Path

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """Pipeline for ingesting documents into vector store"""
    
    def __init__(self, document_processor, embeddings_service, vector_store):
        self.processor = document_processor
        self.embeddings = embeddings_service
        self.vector_store = vector_store
        logger.info("Ingestion pipeline initialized")
    
    async def ingest_directory(self, directory: Path):
        """Ingest all markdown files from directory"""
        logger.info(f"Ingesting documents from: {directory}")
        
        markdown_files = list(directory.glob("**/*.md"))
        logger.info(f"Found {len(markdown_files)} markdown files")
        
        for file_path in markdown_files:
            await self.ingest_file(file_path)
    
    async def ingest_file(self, file_path: Path):
        """Ingest a single file"""
        logger.info(f"Ingesting file: {file_path}")
        
        # Load and process document
        text = file_path.read_text()
        chunks = self.processor.process_text(
            text,
            metadata={"source": str(file_path), "filename": file_path.name}
        )
        
        logger.info(f"Created {len(chunks)} chunks from {file_path.name}")
        
        # TODO: Add to vector store
        # documents = [chunk["text"] for chunk in chunks]
        # metadatas = [chunk["metadata"] for chunk in chunks]
        # ids = [f"{file_path.stem}_{i}" for i in range(len(chunks))]
        # await self.vector_store.add_documents(documents, metadatas, ids)
