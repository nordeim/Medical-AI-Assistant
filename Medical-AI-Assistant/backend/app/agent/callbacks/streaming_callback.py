"""Streaming Callback for Real-time Token Streaming"""
import logging

logger = logging.getLogger(__name__)

class StreamingCallbackHandler:
    """LangChain callback for streaming tokens via WebSocket"""
    
    def __init__(self, websocket_manager, session_id):
        self.ws_manager = websocket_manager
        self.session_id = session_id
        logger.debug(f"Streaming callback initialized for session {session_id}")
    
    async def on_llm_new_token(self, token: str, **kwargs):
        """Called when LLM generates a new token"""
        # TODO: Send token via WebSocket
        await self.ws_manager.send_to_session(self.session_id, {
            "type": "streaming",
            "payload": {"token": token}
        })
    
    async def on_llm_end(self, response, **kwargs):
        """Called when LLM completes"""
        await self.ws_manager.send_to_session(self.session_id, {
            "type": "streaming_complete",
            "payload": {}
        })
