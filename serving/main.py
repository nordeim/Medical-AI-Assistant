"""
Main server entry point for the model serving infrastructure.
Initializes all components and starts the FastAPI application.
"""

import asyncio
import signal
import sys
from pathlib import Path

from fastapi import FastAPI
import structlog

from config.settings import get_settings, validate_config
from config.logging_config import setup_structured_logging, get_logger
from models.base_server import model_registry
from models.concrete_servers import (
    TextGenerationServer, EmbeddingServer, ConversationServer
)
from api.main import app as api_app
from adapters.model_adapters import ModelAdapterFactory, ModelCache


class ModelServingServer:
    """Main server class for managing the model serving infrastructure."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger("server")
        self.model_cache = ModelCache()
        self.is_running = False
        self._shutdown_event = asyncio.Event()
    
    async def initialize(self) -> None:
        """Initialize all server components."""
        try:
            self.logger.info("Initializing model serving server")
            
            # Validate configuration
            config_validation = validate_config()
            if not config_validation["valid"]:
                raise ValueError(f"Configuration validation failed: {config_validation['error']}")
            
            self.logger.info("Configuration validated", **config_validation)
            
            # Create directories
            await self._create_directories()
            
            # Initialize model registry
            await self._initialize_models()
            
            self.logger.info("Server initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Server initialization failed: {e}")
            raise
    
    async def _create_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            Path(self.settings.cache.disk_cache_dir),
            Path(self.settings.logging.log_file).parent if self.settings.logging.log_file else Path("logs"),
            Path(self.settings.medical.audit_log_file).parent if self.settings.medical.audit_log_file else Path("logs")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {directory}")
    
    async def _initialize_models(self) -> None:
        """Initialize and register model servers."""
        try:
            # Text generation model
            text_gen_server = TextGenerationServer(
                model_id="text_generation_v1",
                name="GPT-based Text Generator",
                max_concurrent_requests=self.settings.serving.max_concurrent_requests
            )
            
            # Embedding model
            embedding_server = EmbeddingServer(
                model_id="embedding_v1",
                name="Medical Text Embeddings"
            )
            
            # Conversation model
            conversation_server = ConversationServer(
                model_id="conversation_v1",
                name="Medical AI Assistant",
                max_concurrent_requests=self.settings.serving.max_concurrent_requests
            )
            
            # Register models
            model_registry.register_model(text_gen_server)
            model_registry.register_model(embedding_server)
            model_registry.register_model(conversation_server)
            
            # Initialize models (load them)
            self.logger.info("Loading models...")
            for model_id, model in model_registry.models.items():
                try:
                    await model.initialize()
                    self.logger.info(f"Model loaded successfully: {model_id}")
                except Exception as e:
                    self.logger.error(f"Failed to load model {model_id}: {e}")
                    # Continue with other models
            
            self.logger.info(
                f"Model registry initialized with {len(model_registry.models)} models"
            )
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            raise
    
    async def start(self) -> None:
        """Start the server."""
        try:
            self.is_running = True
            
            # Setup signal handlers
            for sig in [signal.SIGTERM, signal.SIGINT]:
                loop = asyncio.get_event_loop()
                loop.add_signal_handler(
                    sig,
                    lambda: asyncio.create_task(self.shutdown())
                )
            
            self.logger.info(
                "Starting server",
                host=self.settings.serving.host,
                port=self.settings.serving.port,
                environment=self.settings.environment
            )
            
            # Start server using uvicorn
            import uvicorn
            config = uvicorn.Config(
                api_app,
                host=self.settings.serving.host,
                port=self.settings.serving.port,
                reload=self.settings.environment == "development",
                workers=self.settings.serving.workers if self.settings.environment == "production" else 1,
                log_level=self.settings.logging.log_level.lower(),
                access_log=True
            )
            
            server = uvicorn.Server(config)
            
            # Start server in background
            server_task = asyncio.create_task(server.serve())
            
            self.logger.info(f"Server started successfully on {self.settings.serving.host}:{self.settings.serving.port}")
            
            # Wait for shutdown signal
            await self._shutdown_event.wait()
            
            # Shutdown server
            await self.shutdown()
            
        except Exception as e:
            self.logger.error(f"Server startup failed: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the server gracefully."""
        if not self.is_running:
            return
        
        self.logger.info("Initiating server shutdown...")
        self.is_running = False
        
        try:
            # Shutdown model registry
            self.logger.info("Shutting down model registry...")
            for model_id, model in model_registry.models.items():
                try:
                    # Clear cache and cleanup
                    model.cache.clear()
                    if hasattr(model, '_model') and model._model:
                        del model._model
                    if hasattr(model, '_tokenizer') and model._tokenizer:
                        del model._tokenizer
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    self.logger.info(f"Model {model_id} cleaned up")
                except Exception as e:
                    self.logger.error(f"Error cleaning up model {model_id}: {e}")
            
            # Clear model cache
            self.model_cache.clear()
            
            # Set shutdown event
            self._shutdown_event.set()
            
            self.logger.info("Server shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        
        finally:
            sys.exit(0)


async def main():
    """Main entry point."""
    # Setup logging
    setup_structured_logging()
    
    # Create and start server
    server = ModelServingServer()
    
    try:
        await server.initialize()
        await server.start()
    except KeyboardInterrupt:
        await server.shutdown()
    except Exception as e:
        logger = get_logger("main")
        logger.error(f"Server error: {e}")
        await server.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    # Run the server
    asyncio.run(main())