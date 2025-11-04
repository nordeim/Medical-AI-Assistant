import React from 'react';
import { useChat } from '../../contexts/ChatContext';
import { Bot } from 'lucide-react';

const TypingIndicator: React.FC = () => {
  const { isTyping } = useChat();

  if (!isTyping) {
    return null;
  }

  return (
    <div className="flex items-start space-x-3 mr-auto max-w-[80%]">
      {/* Avatar */}
      <div className="flex items-center justify-center w-8 h-8 rounded-full bg-gray-300 text-gray-700 flex-shrink-0">
        <Bot className="w-4 h-4" />
      </div>

      {/* Typing animation */}
      <div className="bg-gray-100 border rounded-lg rounded-tl-lg rounded-tr-lg rounded-br-lg px-4 py-3">
        <div className="flex items-center space-x-1">
          <span className="text-sm text-gray-600 mr-2">AI is thinking</span>
          <div className="flex space-x-1">
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-75" />
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-150" />
          </div>
        </div>
        
        {/* Thinking indicators */}
        <div className="mt-2 text-xs text-gray-500 space-y-1">
          <div className="flex items-center space-x-2">
            <div className="w-1 h-1 bg-blue-400 rounded-full animate-pulse" />
            <span>Processing your symptoms...</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-1 h-1 bg-green-400 rounded-full animate-pulse delay-300" />
            <span>Analyzing medical guidelines...</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-1 h-1 bg-purple-400 rounded-full animate-pulse delay-500" />
            <span>Generating personalized response...</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TypingIndicator;