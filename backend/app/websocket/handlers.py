"""
WebSocket Handlers

WebSocket endpoint handlers and message routing.
"""

import logging
import json
from uuid import UUID, uuid4

from fastapi import WebSocket, WebSocketDisconnect, Query, Depends
from sqlalchemy.orm import Session

from app.database import get_db_context
from app.websocket.manager import ws_manager
from app.schemas.websocket import WebSocketMessage, MessageType
from app.dependencies import verify_websocket_token

logger = logging.getLogger(__name__)


async def handle_websocket_connection(
    websocket: WebSocket,
    session_id: UUID,
    token: str = Query(..., description="JWT access token")
):
    """
    Handle WebSocket connection for patient chat.
    
    Args:
        websocket: WebSocket connection
        session_id: Session UUID
        token: JWT authentication token
    """
    connection_id = str(uuid4())
    
    # Authenticate user
    with get_db_context() as db:
        user = await verify_websocket_token(token, db)
        
        if user is None:
            await websocket.close(code=1008, reason="Authentication failed")
            logger.warning(
                f"WebSocket authentication failed for session {session_id}"
            )
            return
        
        logger.info(
            f"WebSocket authenticated: user={user.id}, session={session_id}"
        )
    
    # Accept connection
    await ws_manager.connect(websocket, connection_id, session_id)
    
    # Send connection success message
    await ws_manager.send_to_connection(connection_id, {
        "type": MessageType.STATUS.value,
        "payload": {
            "status": "connected",
            "session_id": str(session_id)
        }
    })
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
                message = WebSocketMessage(**message_data)
                
                # Route message based on type
                await route_websocket_message(
                    connection_id,
                    session_id,
                    user.id,
                    message
                )
                
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received: {data}")
                await ws_manager.send_to_connection(connection_id, {
                    "type": MessageType.ERROR.value,
                    "payload": {
                        "error": "Invalid message format"
                    }
                })
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                await ws_manager.send_to_connection(connection_id, {
                    "type": MessageType.ERROR.value,
                    "payload": {
                        "error": "Error processing message"
                    }
                })
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
        ws_manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(connection_id)


async def route_websocket_message(
    connection_id: str,
    session_id: UUID,
    user_id: UUID,
    message: WebSocketMessage
):
    """
    Route WebSocket message to appropriate handler.
    
    Args:
        connection_id: WebSocket connection ID
        session_id: Session UUID
        user_id: User UUID
        message: Parsed WebSocket message
    """
    if message.type == MessageType.PING:
        # Handle ping/pong
        await ws_manager.send_to_connection(connection_id, {
            "type": MessageType.PONG.value,
            "payload": {}
        })
    
    elif message.type == MessageType.CHAT:
        # Handle chat message
        # TODO: Integrate with agent orchestrator
        logger.info(
            f"Chat message received from session {session_id}: "
            f"{message.payload.get('content', '')[:50]}"
        )
        
        # Send acknowledgment
        await ws_manager.send_to_connection(connection_id, {
            "type": MessageType.STATUS.value,
            "payload": {
                "status": "message_received",
                "details": "Processing your message..."
            }
        })
        
        # TODO: Trigger agent processing
        # This will be implemented in Phase B (Agent Runtime)
    
    else:
        logger.warning(f"Unknown message type: {message.type}")
