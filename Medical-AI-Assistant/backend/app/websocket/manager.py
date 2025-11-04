"""
WebSocket Connection Manager

Manages active WebSocket connections and message broadcasting.
"""

import logging
from typing import Dict, Optional
from uuid import UUID

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Singleton WebSocket connection manager.
    
    Manages active connections and provides methods for
    sending messages to specific sessions.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._connections: Dict[str, WebSocket] = {}
        self._session_to_connection: Dict[UUID, str] = {}
        self._initialized = True
        
        logger.info("WebSocket manager initialized")
    
    async def connect(
        self,
        websocket: WebSocket,
        connection_id: str,
        session_id: Optional[UUID] = None
    ):
        """
        Register a new WebSocket connection.
        
        Args:
            websocket: WebSocket instance
            connection_id: Unique connection identifier
            session_id: Optional session ID to associate with connection
        """
        await websocket.accept()
        self._connections[connection_id] = websocket
        
        if session_id:
            self._session_to_connection[session_id] = connection_id
        
        logger.info(
            f"WebSocket connection established: {connection_id} "
            f"(session={session_id})"
        )
    
    def disconnect(self, connection_id: str):
        """
        Unregister a WebSocket connection.
        
        Args:
            connection_id: Connection identifier to remove
        """
        if connection_id in self._connections:
            del self._connections[connection_id]
        
        # Remove from session mapping
        session_id_to_remove = None
        for session_id, conn_id in self._session_to_connection.items():
            if conn_id == connection_id:
                session_id_to_remove = session_id
                break
        
        if session_id_to_remove:
            del self._session_to_connection[session_id_to_remove]
        
        logger.info(f"WebSocket connection closed: {connection_id}")
    
    def get_connection(self, connection_id: str) -> Optional[WebSocket]:
        """
        Get WebSocket connection by ID.
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            WebSocket instance if found, None otherwise
        """
        return self._connections.get(connection_id)
    
    def get_connection_by_session(self, session_id: UUID) -> Optional[WebSocket]:
        """
        Get WebSocket connection by session ID.
        
        Args:
            session_id: Session UUID
            
        Returns:
            WebSocket instance if found, None otherwise
        """
        connection_id = self._session_to_connection.get(session_id)
        if connection_id:
            return self._connections.get(connection_id)
        return None
    
    async def send_to_connection(
        self,
        connection_id: str,
        message: dict
    ):
        """
        Send message to specific connection.
        
        Args:
            connection_id: Connection identifier
            message: Message dictionary to send
        """
        websocket = self.get_connection(connection_id)
        if websocket:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(
                    f"Error sending message to {connection_id}: {e}"
                )
                self.disconnect(connection_id)
    
    async def send_to_session(
        self,
        session_id: UUID,
        message: dict
    ):
        """
        Send message to connection associated with session.
        
        Args:
            session_id: Session UUID
            message: Message dictionary to send
        """
        websocket = self.get_connection_by_session(session_id)
        if websocket:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(
                    f"Error sending message to session {session_id}: {e}"
                )
                connection_id = self._session_to_connection.get(session_id)
                if connection_id:
                    self.disconnect(connection_id)
    
    async def broadcast_to_all(self, message: dict):
        """
        Broadcast message to all active connections.
        
        Args:
            message: Message dictionary to broadcast
        """
        disconnected = []
        
        for connection_id, websocket in self._connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(
                    f"Error broadcasting to {connection_id}: {e}"
                )
                disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            self.disconnect(connection_id)
    
    @property
    def active_connections(self) -> int:
        """Get count of active connections"""
        return len(self._connections)


# Global instance
ws_manager = WebSocketManager()
