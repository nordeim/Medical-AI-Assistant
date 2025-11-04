import React from 'react';
import { Card } from '@/components/ui/card';
import { useChat } from '../../contexts/ChatContext';
import { useAuth } from '../../contexts/AuthContext';
import ChatHeader from './ChatHeader';
import ChatMessages from './ChatMessages';
import ChatInput from './ChatInput';
import ConsentModal from '../consent/ConsentModal';
import LoadingScreen from '../ui/LoadingScreen';
import ErrorAlert from '../ui/ErrorAlert';
import ConnectionStatus from '../ui/ConnectionStatus';

const ChatContainer: React.FC = () => {
  const { session, isLoading, consentRequired, error } = useChat();
  const { isAuthenticated, isLoading: authLoading } = useAuth();

  // Show loading screen while authentication is being checked
  if (authLoading || isLoading) {
    return <LoadingScreen message="Initializing chat interface..." />;
  }

  // Show consent modal if required
  if (consentRequired) {
    return <ConsentModal />;
  }

  // Show error if authentication is required but not authenticated
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
        <Card className="w-full max-w-md p-6">
          <div className="text-center">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">
              Authentication Required
            </h2>
            <p className="text-gray-600">
              Please log in to access the medical AI chat interface.
            </p>
          </div>
        </Card>
      </div>
    );
  }

  // Show error if there's an authentication or general error
  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
        <Card className="w-full max-w-md p-6">
          <ErrorAlert 
            title="Chat Error" 
            message={error}
            onRetry={() => window.location.reload()}
          />
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex flex-col">
      {/* Connection Status Indicator */}
      <ConnectionStatus />
      
      {/* Header */}
      <ChatHeader />
      
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col max-w-4xl mx-auto w-full p-4 space-y-4">
        {/* Welcome Message for New Sessions */}
        {!session && (
          <Card className="p-6">
            <div className="text-center space-y-4">
              <div className="w-16 h-16 mx-auto bg-blue-100 rounded-full flex items-center justify-center">
                <svg 
                  className="w-8 h-8 text-blue-600" 
                  fill="none" 
                  stroke="currentColor" 
                  viewBox="0 0 24 24"
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    strokeWidth={2} 
                    d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" 
                  />
                </svg>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900 mb-2">
                  Welcome to Medical AI Assistant
                </h1>
                <p className="text-gray-600 max-w-md mx-auto">
                  I'm here to help you with initial health assessment and triage. 
                  This conversation is for informational purposes and does not replace 
                  professional medical advice.
                </p>
              </div>
            </div>
          </Card>
        )}

        {/* Chat Messages */}
        <Card className="flex-1 min-h-[400px] overflow-hidden">
          <ChatMessages />
        </Card>

        {/* Chat Input */}
        <ChatInput />
      </div>

      {/* Disclaimer Footer */}
      <div className="bg-white/80 backdrop-blur-sm border-t border-gray-200 p-4">
        <div className="max-w-4xl mx-auto">
          <p className="text-xs text-gray-500 text-center">
            <strong>Important:</strong> This AI assistant is for informational and triage purposes only. 
            For medical emergencies, please call 911 or go to your nearest emergency room. 
            This system does not provide medical diagnoses or replace professional medical advice.
          </p>
        </div>
      </div>
    </div>
  );
};

export default ChatContainer;