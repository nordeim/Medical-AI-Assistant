import { WebSocketMessage, ChatWebSocketMessage, Message } from '../types';

// WebSocket Types
type WebSocketEventHandler = (event: WebSocketMessage) => void;
type ConnectionStatusHandler = (status: 'connecting' | 'connected' | 'disconnected' | 'error') => void;

class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectInterval = 1000; // Start with 1 second
  private eventHandlers: Map<string, WebSocketEventHandler[]> = new Map();
  private connectionHandlers: ConnectionStatusHandler[] = [];
  private sessionId: string | null = null;
  private nurseMode = false;
  private isConnecting = false;

  constructor() {
    this.setupEventHandlers();
  }

  private setupEventHandlers() {
    // Handle connection events
    this.on('connection', this.handleConnection.bind(this));
    this.on('disconnection', this.handleDisconnection.bind(this));
    this.on('error', this.handleError.bind(this));
  }

  private handleConnection() {
    this.isConnecting = false;
    this.reconnectAttempts = 0;
    this.reconnectInterval = 1000; // Reset interval
    
    // Join session room if we have one
    if (this.sessionId) {
      this.send({
        type: 'session_join',
        payload: { session_id: this.sessionId },
        timestamp: new Date().toISOString(),
      });
    }

    // Notify connection handlers
    this.connectionHandlers.forEach(handler => handler('connected'));
  }

  private handleDisconnection() {
    this.isConnecting = false;
    this.ws = null;
    
    // Notify connection handlers
    this.connectionHandlers.forEach(handler => handler('disconnected'));
    
    // Attempt to reconnect if not at max attempts
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      setTimeout(() => this.connect(), this.reconnectInterval);
      this.reconnectAttempts++;
      this.reconnectInterval *= 2; // Exponential backoff
    }
  }

  private handleError(event: Event) {
    console.error('WebSocket error:', event);
    this.connectionHandlers.forEach(handler => handler('error'));
    
    if (this.ws?.readyState === WebSocket.CLOSED) {
      this.handleDisconnection();
    }
  }

  connect(sessionId?: string, nurseMode = false): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return Promise.resolve();
    }

    if (this.isConnecting) {
      return Promise.resolve();
    }

    this.isConnecting = true;
    this.sessionId = sessionId || this.sessionId;
    this.nurseMode = nurseMode;

    return new Promise((resolve, reject) => {
      try {
        const wsUrl = this.buildWebSocketUrl();
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
          this.handleConnection();
          resolve();
        };

        this.ws.onclose = () => {
          this.handleDisconnection();
        };

        this.ws.onerror = (error) => {
          this.handleError(error);
          reject(error);
        };

        this.ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
          }
        };

      } catch (error) {
        this.isConnecting = false;
        reject(error);
      }
    });
  }

  private buildWebSocketUrl(): string {
    const baseUrl = import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000';
    const params = new URLSearchParams();
    
    if (this.sessionId) {
      params.append('session_id', this.sessionId);
    }

    if (this.nurseMode) {
      params.append('nurse_mode', 'true');
    }

    return `${baseUrl}/ws${params.toString() ? '?' + params.toString() : ''}`;
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.isConnecting = false;
    this.reconnectAttempts = 0;
    this.connectionHandlers.forEach(handler => handler('disconnected'));
  }

  private handleMessage(message: WebSocketMessage) {
    const handlers = this.eventHandlers.get(message.type) || [];
    handlers.forEach(handler => handler(message));

    // Handle different message types
    switch (message.type) {
      case 'message':
        this.handleChatMessage(message as ChatWebSocketMessage);
        break;
      case 'session_update':
        this.handleSessionUpdate(message);
        break;
      case 'error':
        this.handleErrorMessage(message);
        break;
      case 'consent_required':
        this.handleConsentRequired(message);
        break;
      case 'red_flag_alert':
        this.handleRedFlagAlert(message);
        break;
      case 'new_par':
        this.handleNewPAR(message);
        break;
      case 'par_update':
        this.handlePARUpdate(message);
        break;
      case 'queue_update':
        this.handleQueueUpdate(message);
        break;
    }
  }

  private handleNewPAR(message: WebSocketMessage) {
    console.log('New PAR received:', message);
  }

  private handlePARUpdate(message: WebSocketMessage) {
    console.log('PAR update:', message);
  }

  private handleQueueUpdate(message: WebSocketMessage) {
    console.log('Queue update:', message);
  }

  private handleChatMessage(message: ChatWebSocketMessage) {
    // This will be handled by React components
    console.log('Chat message received:', message);
  }

  private handleSessionUpdate(message: WebSocketMessage) {
    console.log('Session update:', message);
  }

  private handleErrorMessage(message: WebSocketMessage) {
    console.error('WebSocket error message:', message);
  }

  private handleConsentRequired(message: WebSocketMessage) {
    console.log('Consent required:', message);
  }

  private handleRedFlagAlert(message: WebSocketMessage) {
    console.warn('Red flag alert:', message);
  }

  send(message: Partial<WebSocketMessage>) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      const fullMessage: WebSocketMessage = {
        type: message.type || 'message',
        payload: message.payload || {},
        timestamp: message.timestamp || new Date().toISOString(),
      };
      this.ws.send(JSON.stringify(fullMessage));
    } else {
      console.warn('WebSocket not connected. Message not sent:', message);
    }
  }

  // Event handler management
  on(eventType: string, handler: WebSocketEventHandler) {
    if (!this.eventHandlers.has(eventType)) {
      this.eventHandlers.set(eventType, []);
    }
    this.eventHandlers.get(eventType)!.push(handler);
  }

  off(eventType: string, handler: WebSocketEventHandler) {
    const handlers = this.eventHandlers.get(eventType);
    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
  }

  // Connection status handlers
  onConnectionStatusChange(handler: ConnectionStatusHandler) {
    this.connectionHandlers.push(handler);
  }

  offConnectionStatusChange(handler: ConnectionStatusHandler) {
    const index = this.connectionHandlers.indexOf(handler);
    if (index > -1) {
      this.connectionHandlers.splice(index, 1);
    }
  }

  // Utility methods
  get connectionStatus(): 'connecting' | 'connected' | 'disconnected' | 'error' {
    if (!this.ws) return 'disconnected';
    
    switch (this.ws.readyState) {
      case WebSocket.CONNECTING:
        return 'connecting';
      case WebSocket.OPEN:
        return 'connected';
      case WebSocket.CLOSED:
        return 'disconnected';
      case WebSocket.CLOSING:
        return 'connecting';
      default:
        return 'error';
    }
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  setSessionId(sessionId: string) {
    this.sessionId = sessionId;
    if (this.isConnected()) {
      this.send({
        type: 'session_join',
        payload: { session_id: sessionId },
        timestamp: new Date().toISOString(),
      });
    }
  }

  // Cleanup method
  cleanup() {
    this.disconnect();
    this.eventHandlers.clear();
    this.connectionHandlers.length = 0;
  }
}

// Create singleton instance
export const webSocketService = new WebSocketService();
export default webSocketService;