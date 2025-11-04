import React, { useState, useRef, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { useChat } from '../../contexts/ChatContext';
import { Send, Paperclip, Mic, AlertTriangle } from 'lucide-react';
import { cn } from '@/lib/utils';

const ChatInput: React.FC = () => {
  const { 
    currentInput, 
    setInputValue, 
    sendMessage, 
    session, 
    isLoading, 
    isConnected, 
    isStreaming,
    error 
  } = useChat();

  const [isComposing, setIsComposing] = useState(false);
  const [attachments, setAttachments] = useState<File[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }
  }, [currentInput]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!currentInput.trim()) return;
    if (!session) return;
    if (!isConnected) return;
    if (isStreaming) return;

    const message = currentInput.trim();
    setInputValue(''); // Clear input immediately
    
    try {
      await sendMessage(message);
    } catch (error) {
      console.error('Failed to send message:', error);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && !isComposing) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    
    // Validate file types (medical documents only)
    const allowedTypes = [
      'application/pdf',
      'image/jpeg',
      'image/png',
      'image/gif',
      'text/plain'
    ];
    
    const validFiles = files.filter(file => {
      if (!allowedTypes.includes(file.type)) {
        console.warn(`File type not supported: ${file.type}`);
        return false;
      }
      
      // Limit file size to 10MB
      if (file.size > 10 * 1024 * 1024) {
        console.warn(`File too large: ${file.name}`);
        return false;
      }
      
      return true;
    });

    setAttachments(prev => [...prev, ...validFiles]);
    e.target.value = ''; // Reset input
  };

  const removeAttachment = (index: number) => {
    setAttachments(prev => prev.filter((_, i) => i !== index));
  };

  const isInputDisabled = !session || !isConnected || isStreaming || isLoading;

  return (
    <Card className="bg-white/90 backdrop-blur-sm border-0 shadow-sm">
      <div className="p-4">
        {/* Error Display */}
        {error && (
          <div className="mb-3 p-3 bg-red-50 border border-red-200 rounded-lg flex items-center space-x-2">
            <AlertTriangle className="w-4 h-4 text-red-600" />
            <p className="text-sm text-red-700">{error}</p>
          </div>
        )}

        {/* Connection Status Warning */}
        {!isConnected && (
          <div className="mb-3 p-3 bg-yellow-50 border border-yellow-200 rounded-lg flex items-center space-x-2">
            <AlertTriangle className="w-4 h-4 text-yellow-600" />
            <p className="text-sm text-yellow-700">
              Connection lost. Attempting to reconnect...
            </p>
          </div>
        )}

        {/* File Attachments */}
        {attachments.length > 0 && (
          <div className="mb-3 space-y-2">
            {attachments.map((file, index) => (
              <div 
                key={index}
                className="flex items-center justify-between p-2 bg-gray-50 rounded-lg"
              >
                <div className="flex items-center space-x-2">
                  <Paperclip className="w-4 h-4 text-gray-500" />
                  <span className="text-sm text-gray-700">{file.name}</span>
                  <span className="text-xs text-gray-500">
                    ({(file.size / 1024).toFixed(1)} KB)
                  </span>
                </div>
                <button
                  onClick={() => removeAttachment(index)}
                  className="text-xs text-red-600 hover:text-red-700"
                >
                  Remove
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Main Input Form */}
        <form onSubmit={handleSubmit} className="space-y-3">
          <div className="relative">
            <Textarea
              ref={textareaRef}
              value={currentInput}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              onCompositionStart={() => setIsComposing(true)}
              onCompositionEnd={() => setIsComposing(false)}
              placeholder={
                !session ? "Start a new session to begin..." :
                !isConnected ? "Reconnecting..." :
                "Describe your symptoms or health concerns..."
              }
              disabled={isInputDisabled}
              className={cn(
                "min-h-[50px] max-h-[120px] resize-none pr-12",
                isInputDisabled && "opacity-50 cursor-not-allowed"
              )}
              maxLength={1000}
            />

            {/* Character Count */}
            <div className="absolute bottom-2 right-2 text-xs text-gray-400">
              {currentInput.length}/1000
            </div>
          </div>

          {/* Input Actions */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              {/* File Upload Button */}
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => fileInputRef.current?.click()}
                disabled={isInputDisabled}
                className="text-xs"
              >
                <Paperclip className="w-4 h-4 mr-1" />
                Attach
              </Button>

              {/* Voice Input Button (Future Feature) */}
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={true}
                className="text-xs opacity-50"
                title="Voice input coming soon"
              >
                <Mic className="w-4 h-4 mr-1" />
                Voice
              </Button>

              {/* Safety Reminder */}
              <div className="text-xs text-gray-500 hidden md:block">
                For medical emergencies, call 911
              </div>
            </div>

            {/* Send Button */}
            <Button
              type="submit"
              disabled={
                isInputDisabled || 
                !currentInput.trim() || 
                currentInput.length > 1000
              }
              className="bg-blue-600 hover:bg-blue-700 text-white"
            >
              <Send className="w-4 h-4 mr-1" />
              Send
            </Button>
          </div>

          {/* Hidden File Input */}
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.jpg,.jpeg,.png,.gif,.txt"
            onChange={handleFileUpload}
            className="hidden"
          />
        </form>

        {/* Input Guidelines */}
        <div className="mt-3 p-3 bg-blue-50 rounded-lg">
          <p className="text-xs text-blue-700">
            <strong>Tips for better assistance:</strong> Be specific about your symptoms, 
            when they started, and any factors that make them better or worse. 
            Mention any medications you're taking and relevant medical history.
          </p>
        </div>

        {/* Streaming Indicator */}
        {isStreaming && (
          <div className="mt-3 p-3 bg-green-50 rounded-lg flex items-center space-x-2">
            <div className="flex space-x-1">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-bounce" />
              <div className="w-2 h-2 bg-green-500 rounded-full animate-bounce delay-75" />
              <div className="w-2 h-2 bg-green-500 rounded-full animate-bounce delay-150" />
            </div>
            <p className="text-sm text-green-700">
              AI is analyzing your message and preparing a response...
            </p>
          </div>
        )}
      </div>
    </Card>
  );
};

export default ChatInput;