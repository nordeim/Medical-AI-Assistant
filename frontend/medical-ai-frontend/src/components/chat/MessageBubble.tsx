import React from 'react';
import { cn } from '@/lib/utils';
import { ChatMessage } from '../../types';
import { User, Bot, AlertCircle, CheckCircle } from 'lucide-react';

interface MessageBubbleProps {
  message: ChatMessage;
  isStreaming?: boolean;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message, isStreaming }) => {
  const isUser = message.sender_type === 'patient';
  const isAgent = message.sender_type === 'agent';
  const isSystem = message.sender_type === 'system';
  const isStreamingMessage = isStreaming || message.metadata?.isStreaming;

  const getMessageStyles = () => {
    if (isUser) {
      return {
        container: 'ml-auto max-w-[80%]',
        bubble: 'bg-blue-600 text-white rounded-tl-lg rounded-tr-lg rounded-bl-lg',
        icon: <User className="w-4 h-4" />,
      };
    }
    
    if (isAgent) {
      return {
        container: 'mr-auto max-w-[80%]',
        bubble: 'bg-gray-100 text-gray-900 rounded-tl-lg rounded-tr-lg rounded-br-lg border',
        icon: <Bot className="w-4 h-4" />,
      };
    }
    
    if (isSystem) {
      return {
        container: 'mx-auto max-w-[90%]',
        bubble: 'bg-yellow-50 text-yellow-800 rounded-lg border border-yellow-200',
        icon: <AlertCircle className="w-4 h-4" />,
      };
    }

    return {
      container: 'mr-auto max-w-[80%]',
      bubble: 'bg-gray-100 text-gray-900',
      icon: <Bot className="w-4 h-4" />,
    };
  };

  const styles = getMessageStyles();
  const timeString = new Date(message.timestamp).toLocaleTimeString([], { 
    hour: '2-digit', 
    minute: '2-digit' 
  });

  return (
    <div className={cn('flex flex-col space-y-1', styles.container)}>
      {/* Message Header */}
      <div className="flex items-center space-x-2 px-1">
        <div className={cn(
          'flex items-center justify-center w-6 h-6 rounded-full',
          isUser ? 'bg-blue-500 text-white' :
          isAgent ? 'bg-gray-300 text-gray-700' :
          'bg-yellow-500 text-white'
        )}>
          {styles.icon}
        </div>
        
        <div className="text-xs text-gray-500">
          <span className="font-medium">
            {isUser ? 'You' : 
             isAgent ? 'Medical AI' : 
             isSystem ? 'System' : 'Unknown'}
          </span>
          <span className="ml-1 text-gray-400">•</span>
          <span className="ml-1">{timeString}</span>
        </div>

        {/* Streaming indicator */}
        {isStreamingMessage && (
          <div className="flex items-center space-x-1">
            <div className="flex space-x-1">
              <div className="w-1 h-1 bg-gray-400 rounded-full animate-pulse" />
              <div className="w-1 h-1 bg-gray-400 rounded-full animate-pulse delay-75" />
              <div className="w-1 h-1 bg-gray-400 rounded-full animate-pulse delay-150" />
            </div>
          </div>
        )}

        {/* Status indicators */}
        {isAgent && message.metadata?.safetyFlagged && (
          <div className="flex items-center space-x-1 text-xs text-yellow-600">
            <AlertCircle className="w-3 h-3" />
            <span>Flagged for review</span>
          </div>
        )}

        {isStreamingMessage && (
          <div className="flex items-center space-x-1 text-xs text-blue-600">
            <CheckCircle className="w-3 h-3" />
            <span>Live</span>
          </div>
        )}
      </div>

      {/* Message Content */}
      <div className={cn(
        'px-4 py-3 break-words',
        styles.bubble
      )}>
        {/* System message styling */}
        {isSystem ? (
          <div className="flex items-start space-x-2">
            <AlertCircle className="w-5 h-5 mt-0.5 flex-shrink-0" />
            <div>
              <p className="font-medium mb-1">Important Notice</p>
              <div className="text-sm whitespace-pre-wrap">{message.content}</div>
            </div>
          </div>
        ) : (
          <div className="text-sm whitespace-pre-wrap leading-relaxed">
            {message.content}
            
            {/* Add ellipsis for streaming messages */}
            {isStreamingMessage && (
              <span className="inline-block w-2 h-4 bg-current opacity-75 animate-pulse ml-1" />
            )}
          </div>
        )}

        {/* Additional metadata for debug/info */}
        {message.metadata && Object.keys(message.metadata).length > 0 && !isStreaming && (
          <details className="mt-2 pt-2 border-t border-gray-200">
            <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700">
              Message details
            </summary>
            <pre className="text-xs text-gray-400 mt-1 overflow-x-auto">
              {JSON.stringify(message.metadata, null, 2)}
            </pre>
          </details>
        )}
      </div>

      {/* Message Actions */}
      <div className="flex items-center justify-between px-1">
        <div className="text-xs text-gray-400">
          {isAgent && 'AI-generated response'}
        </div>
        
        <div className="flex items-center space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            className="text-xs text-gray-400 hover:text-gray-600 p-1 rounded"
            onClick={() => {
              navigator.clipboard.writeText(message.content);
              // You might want to show a toast notification here
            }}
            title="Copy message"
          >
            Copy
          </button>
          
          {isAgent && (
            <button
              className="text-xs text-gray-400 hover:text-gray-600 p-1 rounded"
              title="Report message"
              onClick={() => {
                // This would open a report dialog
                console.log('Report message:', message.id);
              }}
            >
              Report
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default MessageBubble;