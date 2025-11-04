import React, { createContext, useContext, useReducer, useEffect, ReactNode, useCallback } from 'react';
import { Session, Message, ConsentInfo, ChatMessage, WebSocketMessage } from '../types';
import { webSocketService } from '../services/websocket';
import { apiService } from '../services/api';
import { useAuth } from './AuthContext';

interface ChatState {
  session: Session | null;
  messages: ChatMessage[];
  isConnected: boolean;
  isTyping: boolean;
  isLoading: boolean;
  error: string | null;
  consentRequired: boolean;
  consentInfo: ConsentInfo | null;
  currentInput: string;
  isStreaming: boolean;
}

type ChatAction =
  | { type: 'CHAT_START' }
  | { type: 'SESSION_CREATED'; payload: Session }
  | { type: 'SESSION_LOADED'; payload: Session }
  | { type: 'MESSAGES_LOADED'; payload: Message[] }
  | { type: 'MESSAGE_RECEIVED'; payload: ChatMessage }
  | { type: 'MESSAGE_SENT'; payload: ChatMessage }
  | { type: 'STREAMING_START' }
  | { type: 'STREAMING_END' }
  | { type: 'TYPING_START' }
  | { type: 'TYPING_END' }
  | { type: 'CONNECTION_STATUS'; payload: 'connected' | 'disconnected' | 'connecting' | 'error' }
  | { type: 'CONSENT_REQUIRED'; payload: ConsentInfo }
  | { type: 'CONSENT_ACCEPTED' }
  | { type: 'INPUT_CHANGE'; payload: string }
  | { type: 'CHAT_ERROR'; payload: string }
  | { type: 'CLEAR_ERROR' }
  | { type: 'RESET_CHAT' };

interface ChatContextType extends ChatState {
  createSession: () => Promise<boolean>;
  loadSession: (sessionId: string) => Promise<boolean>;
  sendMessage: (content: string) => Promise<boolean>;
  setInputValue: (value: string) => void;
  clearError: () => void;
  handleConsent: (responses: Record<string, any>) => Promise<boolean>;
  resetChat: () => void;
  connect: () => Promise<void>;
  disconnect: () => void;
}

const initialState: ChatState = {
  session: null,
  messages: [],
  isConnected: false,
  isTyping: false,
  isLoading: false,
  error: null,
  consentRequired: false,
  consentInfo: null,
  currentInput: '',
  isStreaming: false,
};

const chatReducer = (state: ChatState, action: ChatAction): ChatState => {
  switch (action.type) {
    case 'CHAT_START':
      return {
        ...state,
        isLoading: true,
        error: null,
      };

    case 'SESSION_CREATED':
    case 'SESSION_LOADED':
      return {
        ...state,
        session: action.payload,
        isLoading: false,
        error: null,
      };

    case 'MESSAGES_LOADED':
      return {
        ...state,
        messages: action.payload.map(msg => ({
          ...msg,
          sender: msg.sender_type as 'patient' | 'agent' | 'system',
          isUser: msg.sender_type === 'patient',
          text: msg.content,
          timestamp: new Date(msg.timestamp),
        })),
        isLoading: false,
      };

    case 'MESSAGE_RECEIVED':
    case 'MESSAGE_SENT':
      return {
        ...state,
        messages: [...state.messages, action.payload],
        isLoading: false,
      };

    case 'STREAMING_START':
      return {
        ...state,
        isStreaming: true,
        isTyping: true,
      };

    case 'STREAMING_END':
      return {
        ...state,
        isStreaming: false,
        isTyping: false,
      };

    case 'TYPING_START':
      return {
        ...state,
        isTyping: true,
      };

    case 'TYPING_END':
      return {
        ...state,
        isTyping: false,
      };

    case 'CONNECTION_STATUS':
      return {
        ...state,
        isConnected: action.payload === 'connected',
      };

    case 'CONSENT_REQUIRED':
      return {
        ...state,
        consentRequired: true,
        consentInfo: action.payload,
      };

    case 'CONSENT_ACCEPTED':
      return {
        ...state,
        consentRequired: false,
        consentInfo: null,
      };

    case 'INPUT_CHANGE':
      return {
        ...state,
        currentInput: action.payload,
      };

    case 'CHAT_ERROR':
      return {
        ...state,
        error: action.payload,
        isLoading: false,
        isTyping: false,
        isStreaming: false,
      };

    case 'CLEAR_ERROR':
      return {
        ...state,
        error: null,
      };

    case 'RESET_CHAT':
      return {
        ...initialState,
        isConnected: state.isConnected, // Preserve connection status
      };

    default:
      return state;
  }
};

const ChatContext = createContext<ChatContextType | undefined>(undefined);

interface ChatProviderProps {
  children: ReactNode;
}

export const ChatProvider: React.FC<ChatProviderProps> = ({ children }) => {
  const [state, dispatch] = useReducer(chatReducer, initialState);
  const { user, isAuthenticated } = useAuth();

  // WebSocket event handlers
  const handleWebSocketMessage = useCallback((message: WebSocketMessage) => {
    switch (message.type) {
      case 'message':
        const chatMessage: ChatMessage = {
          id: message.payload.message_id,
          session_id: message.payload.session_id,
          sender_type: message.payload.sender_type,
          content: message.payload.content,
          sender: message.payload.sender_type as 'patient' | 'agent' | 'system',
          isUser: message.payload.sender_type === 'patient',
          text: message.payload.content,
          timestamp: new Date(message.payload.timestamp),
          metadata: {
            isStreaming: message.payload.is_streaming,
            ...message.payload.metadata,
          },
        };

        if (message.payload.is_streaming) {
          dispatch({ type: 'STREAMING_END' });
          // Update the last message instead of adding a new one
          dispatch({
            type: 'MESSAGE_RECEIVED',
            payload: chatMessage,
          });
        } else {
          dispatch({ type: 'MESSAGE_RECEIVED', payload: chatMessage });
        }
        break;

      case 'typing':
        if (message.payload.is_typing) {
          dispatch({ type: 'TYPING_START' });
        } else {
          dispatch({ type: 'TYPING_END' });
        }
        break;

      case 'session_update':
        // Handle session updates
        break;

      case 'error':
        dispatch({ type: 'CHAT_ERROR', payload: message.payload.message || 'An error occurred' });
        break;

      case 'consent_required':
        dispatch({ type: 'CONSENT_REQUIRED', payload: message.payload.consent_info });
        break;

      case 'red_flag_alert':
        // Handle red flag alerts
        console.warn('Red flag alert:', message.payload);
        break;
    }
  }, []);

  const handleConnectionStatus = useCallback((status: 'connecting' | 'connected' | 'disconnected' | 'error') => {
    dispatch({ type: 'CONNECTION_STATUS', payload: status });
  }, []);

  // Set up WebSocket event listeners
  useEffect(() => {
    webSocketService.on('message', handleWebSocketMessage);
    webSocketService.on('typing', handleWebSocketMessage);
    webSocketService.on('session_update', handleWebSocketMessage);
    webSocketService.on('error', handleWebSocketMessage);
    webSocketService.onConnectionStatusChange(handleConnectionStatus);

    return () => {
      webSocketService.off('message', handleWebSocketMessage);
      webSocketService.off('typing', handleWebSocketMessage);
      webSocketService.off('session_update', handleWebSocketMessage);
      webSocketService.off('error', handleWebSocketMessage);
      webSocketService.offConnectionStatusChange(handleConnectionStatus);
    };
  }, [handleWebSocketMessage, handleConnectionStatus]);

  // Connect to WebSocket when user is authenticated
  useEffect(() => {
    if (isAuthenticated && user) {
      connect();
    }
  }, [isAuthenticated, user]);

  const createSession = async (): Promise<boolean> => {
    dispatch({ type: 'CHAT_START' });

    try {
      const response = await apiService.createSession();
      
      if (response.success && response.data) {
        dispatch({ type: 'SESSION_CREATED', payload: response.data });
        webSocketService.setSessionId(response.data.id);
        return true;
      } else {
        const errorMessage = response.error?.message || 'Failed to create session';
        dispatch({ type: 'CHAT_ERROR', payload: errorMessage });
        return false;
      }
    } catch (error: any) {
      const errorMessage = error.message || 'Failed to create session';
      dispatch({ type: 'CHAT_ERROR', payload: errorMessage });
      return false;
    }
  };

  const loadSession = async (sessionId: string): Promise<boolean> => {
    dispatch({ type: 'CHAT_START' });

    try {
      const response = await apiService.getSession(sessionId);
      
      if (response.success && response.data) {
        dispatch({ type: 'SESSION_LOADED', payload: response.data });
        
        // Load messages for this session
        const messagesResponse = await apiService.getSessionMessages(sessionId);
        if (messagesResponse.success && messagesResponse.data) {
          dispatch({ type: 'MESSAGES_LOADED', payload: messagesResponse.data });
        }

        webSocketService.setSessionId(sessionId);
        return true;
      } else {
        const errorMessage = response.error?.message || 'Failed to load session';
        dispatch({ type: 'CHAT_ERROR', payload: errorMessage });
        return false;
      }
    } catch (error: any) {
      const errorMessage = error.message || 'Failed to load session';
      dispatch({ type: 'CHAT_ERROR', payload: errorMessage });
      return false;
    }
  };

  const sendMessage = async (content: string): Promise<boolean> => {
    if (!state.session) {
      dispatch({ type: 'CHAT_ERROR', payload: 'No active session' });
      return false;
    }

    // Add user message to UI immediately
    const userMessage: ChatMessage = {
      id: `temp-${Date.now()}`,
      session_id: state.session.id,
      sender_type: 'patient',
      content,
      sender: 'patient',
      isUser: true,
      text: content,
      timestamp: new Date(),
    };

    dispatch({ type: 'MESSAGE_SENT', payload: userMessage });
    dispatch({ type: 'INPUT_CHANGE', payload: '' });

    try {
      const response = await apiService.sendMessage(state.session.id, content);
      
      if (response.success && response.data) {
        // Send via WebSocket as well for real-time delivery
        webSocketService.send({
          type: 'message',
          payload: {
            content,
            session_id: state.session.id,
          },
          timestamp: new Date().toISOString(),
        });

        dispatch({ type: 'STREAMING_START' });
        return true;
      } else {
        const errorMessage = response.error?.message || 'Failed to send message';
        dispatch({ type: 'CHAT_ERROR', payload: errorMessage });
        return false;
      }
    } catch (error: any) {
      const errorMessage = error.message || 'Failed to send message';
      dispatch({ type: 'CHAT_ERROR', payload: errorMessage });
      return false;
    }
  };

  const handleConsent = async (responses: Record<string, any>): Promise<boolean> => {
    if (!state.consentInfo || !user) {
      return false;
    }

    try {
      const consentResponse = {
        id: `consent_${Date.now()}`, // Generate temporary ID
        consent_id: state.consentInfo.id,
        user_id: user.id,
        responses,
        signed_at: new Date().toISOString(),
        ip_address: '127.0.0.1', // In real app, get from server
        user_agent: navigator.userAgent,
      };

      const response = await apiService.submitConsent(consentResponse);
      
      if (response.success) {
        dispatch({ type: 'CONSENT_ACCEPTED' });
        return true;
      } else {
        const errorMessage = response.error?.message || 'Failed to submit consent';
        dispatch({ type: 'CHAT_ERROR', payload: errorMessage });
        return false;
      }
    } catch (error: any) {
      const errorMessage = error.message || 'Failed to submit consent';
      dispatch({ type: 'CHAT_ERROR', payload: errorMessage });
      return false;
    }
  };

  const connect = async (): Promise<void> => {
    if (state.session) {
      await webSocketService.connect(state.session.id);
    } else {
      await webSocketService.connect();
    }
  };

  const disconnect = (): void => {
    webSocketService.disconnect();
  };

  const setInputValue = (value: string): void => {
    dispatch({ type: 'INPUT_CHANGE', payload: value });
  };

  const clearError = (): void => {
    dispatch({ type: 'CLEAR_ERROR' });
  };

  const resetChat = (): void => {
    dispatch({ type: 'RESET_CHAT' });
  };

  const value: ChatContextType = {
    ...state,
    createSession,
    loadSession,
    sendMessage,
    setInputValue,
    clearError,
    handleConsent,
    resetChat,
    connect,
    disconnect,
  };

  return (
    <ChatContext.Provider value={value}>
      {children}
    </ChatContext.Provider>
  );
};

export const useChat = (): ChatContextType => {
  const context = useContext(ChatContext);
  if (context === undefined) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
};