"""
Message Service

Business logic for message management.
"""

import logging
from typing import List, Optional
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.models.message import Message as MessageModel, MessageRole
from app.models.session import Session as SessionModel
from app.schemas.message import MessageCreate

logger = logging.getLogger(__name__)


class MessageService:
    """Service for managing messages"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_message(
        self,
        session_id: UUID,
        message_data: MessageCreate,
        role: MessageRole
    ) -> MessageModel:
        """
        Create a new message in a session.
        
        Args:
            session_id: Session UUID
            message_data: Message content and metadata
            role: Message role (user, assistant, system)
            
        Returns:
            MessageModel: Created message
        """
        # Get current message count for sequence number
        sequence_number = self.db.query(MessageModel).filter(
            MessageModel.session_id == session_id
        ).count() + 1
        
        message = MessageModel(
            session_id=session_id,
            role=role,
            content=message_data.content,
            sequence_number=sequence_number,
            metadata=message_data.metadata or {}
        )
        
        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        
        logger.info(
            f"Created message {message.id} (role={role}) in session {session_id}"
        )
        return message
    
    def get_message(self, message_id: UUID) -> Optional[MessageModel]:
        """
        Get a message by ID.
        
        Args:
            message_id: Message UUID
            
        Returns:
            MessageModel: Message if found, None otherwise
        """
        return self.db.query(MessageModel).filter(
            MessageModel.id == message_id
        ).first()
    
    def get_session_messages(
        self,
        session_id: UUID,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[MessageModel]:
        """
        Get all messages for a session.
        
        Args:
            session_id: Session UUID
            limit: Maximum number of messages to return
            offset: Number of messages to skip
            
        Returns:
            List of messages ordered by sequence number
        """
        query = self.db.query(MessageModel).filter(
            MessageModel.session_id == session_id
        ).order_by(MessageModel.sequence_number)
        
        if limit is not None:
            query = query.limit(limit).offset(offset)
        
        return query.all()
    
    def get_latest_messages(
        self,
        session_id: UUID,
        count: int = 10
    ) -> List[MessageModel]:
        """
        Get the latest N messages from a session.
        
        Args:
            session_id: Session UUID
            count: Number of messages to retrieve
            
        Returns:
            List of latest messages
        """
        messages = self.db.query(MessageModel).filter(
            MessageModel.session_id == session_id
        ).order_by(desc(MessageModel.sequence_number)).limit(count).all()
        
        # Return in chronological order
        return list(reversed(messages))
    
    def count_session_messages(self, session_id: UUID) -> int:
        """
        Count total messages in a session.
        
        Args:
            session_id: Session UUID
            
        Returns:
            int: Total message count
        """
        return self.db.query(MessageModel).filter(
            MessageModel.session_id == session_id
        ).count()
    
    def get_conversation_history(
        self,
        session_id: UUID,
        format: str = "langchain"
    ) -> List[dict]:
        """
        Get conversation history formatted for LLM consumption.
        
        Args:
            session_id: Session UUID
            format: Output format ('langchain' or 'openai')
            
        Returns:
            List of message dictionaries
        """
        messages = self.get_session_messages(session_id)
        
        if format == "langchain":
            return [
                {
                    "role": msg.role.value,
                    "content": msg.content
                }
                for msg in messages
            ]
        elif format == "openai":
            return [
                {
                    "role": msg.role.value if msg.role != MessageRole.ASSISTANT else "assistant",
                    "content": msg.content
                }
                for msg in messages
            ]
        else:
            raise ValueError(f"Unknown format: {format}")
