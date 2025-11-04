"""Audit Callback for Logging"""
import logging

logger = logging.getLogger(__name__)

class AuditCallbackHandler:
    """LangChain callback for audit logging"""
    
    def __init__(self, audit_service):
        self.audit_service = audit_service
        logger.debug("Audit callback initialized")
    
    async def on_tool_start(self, tool, input_str, **kwargs):
        """Log tool usage"""
        logger.debug(f"Tool started: {tool} with input: {input_str[:50]}")
    
    async def on_llm_start(self, llm, prompts, **kwargs):
        """Log LLM call"""
        logger.debug("LLM call started")
    
    async def on_llm_end(self, response, **kwargs):
        """Log completion"""
        logger.debug("LLM call completed")
