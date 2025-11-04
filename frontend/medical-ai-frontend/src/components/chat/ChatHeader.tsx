import React from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { useChat } from '../../contexts/ChatContext';
import { useAuth } from '../../contexts/AuthContext';
import { Activity, Clock, User, AlertTriangle } from 'lucide-react';

const ChatHeader: React.FC = () => {
  const { session, isConnected, isTyping } = useChat();
  const { user } = useAuth();

  const formatSessionTime = (createdAt: string) => {
    const date = new Date(createdAt);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getSessionDuration = (createdAt: string) => {
    const start = new Date(createdAt);
    const now = new Date();
    const diffMs = now.getTime() - start.getTime();
    const diffMinutes = Math.floor(diffMs / 60000);
    const hours = Math.floor(diffMinutes / 60);
    const minutes = diffMinutes % 60;
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  };

  const getConnectionStatusColor = () => {
    if (!isConnected) return 'bg-gray-500';
    if (isTyping) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  const getConnectionStatusText = () => {
    if (!isConnected) return 'Disconnected';
    if (isTyping) return 'Assistant is typing...';
    return 'Connected';
  };

  return (
    <Card className="bg-white/90 backdrop-blur-sm border-0 shadow-sm sticky top-0 z-10">
      <div className="px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Left: Session Info */}
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                <Activity className="w-5 h-5 text-blue-600" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-gray-900">
                  Medical AI Assistant
                </h2>
                <p className="text-sm text-gray-500">
                  Health Assessment & Triage Support
                </p>
              </div>
            </div>
          </div>

          {/* Center: Session Status */}
          {session && (
            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <Clock className="w-4 h-4" />
                <span>Started {formatSessionTime(session.created_at)}</span>
                <Badge variant="outline" className="text-xs">
                  {getSessionDuration(session.created_at)}
                </Badge>
              </div>
              
              <Badge 
                variant={session.status === 'active' ? 'default' : 'secondary'}
                className="text-xs"
              >
                {session.status}
              </Badge>
            </div>
          )}

          {/* Right: User & Connection */}
          <div className="flex items-center space-x-4">
            {user && (
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <User className="w-4 h-4" />
                <span>{user.first_name || user.email}</span>
                <Badge variant="outline" className="text-xs">
                  {user.role}
                </Badge>
              </div>
            )}

            {/* Connection Status */}
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${getConnectionStatusColor()}`} />
              <span className="text-sm text-gray-600">
                {getConnectionStatusText()}
              </span>
            </div>
          </div>
        </div>

        {/* Session ID and Quick Actions */}
        {session && (
          <div className="mt-3 flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="text-xs text-gray-500">
                Session ID: <code className="bg-gray-100 px-1 py-0.5 rounded">{session.id}</code>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              {session.status === 'active' && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    // This would trigger session end logic
                    console.log('End session clicked');
                  }}
                  className="text-xs"
                >
                  End Session
                </Button>
              )}
              
              <Button
                variant="ghost"
                size="sm"
                className="text-xs"
                onClick={() => {
                  // This would show session info
                  console.log('Session info clicked');
                }}
              >
                <AlertTriangle className="w-4 h-4 mr-1" />
                Safety Info
              </Button>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

export default ChatHeader;