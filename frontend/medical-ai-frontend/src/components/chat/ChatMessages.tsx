import React, { useEffect, useRef } from 'react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useChat } from '../../contexts/ChatContext';
import MessageBubble from './MessageBubble';
import TypingIndicator from './TypingIndicator';
import WelcomeMessage from './WelcomeMessage';
import { MessageSquare } from 'lucide-react';

const ChatMessages: React.FC = () => {
  const { messages, session, isLoading } = useChat();
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Show loading state for messages
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex items-center space-x-2 text-gray-500">
          <MessageSquare className="w-5 h-5 animate-pulse" />
          <span>Loading conversation...</span>
        </div>
      </div>
    );
  }

  // Show welcome message if no session or messages
  if (!session || messages.length === 0) {
    return <WelcomeMessage />;
  }

  return (
    <div className="flex flex-col h-full">
      <ScrollArea 
        ref={scrollAreaRef}
        className="flex-1 p-4"
        style={{ height: 'calc(100vh - 300px)' }}
      >
        <div className="space-y-4 max-w-3xl mx-auto">
          {/* Welcome message for existing sessions */}
          {messages.length > 0 && messages[0].sender_type === 'system' && (
            <div className="text-center mb-6">
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h3 className="font-medium text-blue-900 mb-2">
                  Medical AI Assessment Session
                </h3>
                <p className="text-sm text-blue-700">
                  This conversation helps me understand your health concerns for initial assessment and triage. 
                  Please answer questions thoroughly and honestly.
                </p>
              </div>
            </div>
          )}

          {/* Render messages */}
          {messages.map((message, index) => (
            <MessageBubble
              key={message.id || `temp-${index}`}
              message={message}
              isStreaming={message.metadata?.isStreaming}
            />
          ))}

          {/* Typing indicator */}
          <TypingIndicator />

          {/* Invisible element for auto-scrolling */}
          <div ref={messagesEndRef} />
        </div>
      </ScrollArea>

      {/* Message Actions */}
      {messages.length > 0 && (
        <div className="border-t bg-gray-50 px-4 py-3">
          <div className="max-w-3xl mx-auto flex items-center justify-between">
            <div className="text-xs text-gray-500">
              {messages.length} message{messages.length !== 1 ? 's' : ''}
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => {
                  // Scroll to top
                  scrollAreaRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
                }}
                className="text-xs text-blue-600 hover:text-blue-700"
              >
                ↑ Back to top
              </button>
              <span className="text-xs text-gray-300">•</span>
              <button
                onClick={() => {
                  // Clear conversation (with confirmation)
                  if (window.confirm('Clear conversation history? This cannot be undone.')) {
                    // This would trigger clear conversation logic
                    console.log('Clear conversation clicked');
                  }
                }}
                className="text-xs text-red-600 hover:text-red-700"
              >
                Clear chat
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatMessages;