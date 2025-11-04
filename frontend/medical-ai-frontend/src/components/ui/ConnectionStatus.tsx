import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { useChat } from '../../contexts/ChatContext';
import { webSocketService } from '../../services/websocket';
import { Wifi, WifiOff, Loader2, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

const ConnectionStatus: React.FC = () => {
  const { isConnected } = useChat();
  const isConnecting = webSocketService.connectionStatus === 'connecting';
  const [showStatus, setShowStatus] = useState(false);

  // Show connection status changes briefly
  useEffect(() => {
    setShowStatus(true);
    const timer = setTimeout(() => setShowStatus(false), 3000);
    return () => clearTimeout(timer);
  }, [isConnected, isConnecting]);

  const getStatusConfig = () => {
    if (isConnecting) {
      return {
        icon: <Loader2 className="w-4 h-4 animate-spin" />,
        color: 'text-yellow-600 bg-yellow-50 border-yellow-200',
        text: 'Connecting...',
        dotColor: 'bg-yellow-500'
      };
    }
    
    if (isConnected) {
      return {
        icon: <Wifi className="w-4 h-4" />,
        color: 'text-green-600 bg-green-50 border-green-200',
        text: 'Connected',
        dotColor: 'bg-green-500'
      };
    }
    
    return {
      icon: <WifiOff className="w-4 h-4" />,
      color: 'text-red-600 bg-red-50 border-red-200',
      text: 'Disconnected',
      dotColor: 'bg-red-500'
    };
  };

  const config = getStatusConfig();

  if (!showStatus) {
    return null;
  }

  return (
    <div className="fixed top-4 right-4 z-50">
      <Card className={cn(
        "px-3 py-2 border shadow-lg backdrop-blur-sm",
        config.color
      )}>
        <div className="flex items-center space-x-2">
          <div className={cn("w-2 h-2 rounded-full", config.dotColor)} />
          {config.icon}
          <span className="text-sm font-medium">{config.text}</span>
          
          {!isConnected && !isConnecting && (
            <button
              onClick={() => setShowStatus(false)}
              className="ml-1 text-xs opacity-70 hover:opacity-100"
            >
              ×
            </button>
          )}
        </div>
      </Card>
    </div>
  );
};

// Compact connection indicator (always visible)
export const CompactConnectionStatus: React.FC = () => {
  const { isConnected } = useChat();
  const isConnecting = webSocketService.connectionStatus === 'connecting';

  const getStatusConfig = () => {
    if (isConnecting) {
      return {
        dotColor: 'bg-yellow-500 animate-pulse',
        tooltip: 'Connecting...'
      };
    }
    
    if (isConnected) {
      return {
        dotColor: 'bg-green-500',
        tooltip: 'Connected to AI assistant'
      };
    }
    
    return {
      dotColor: 'bg-red-500',
      tooltip: 'Connection lost'
    };
  };

  const config = getStatusConfig();

  return (
    <div 
      className="fixed bottom-4 right-4 z-40 cursor-pointer"
      title={config.tooltip}
    >
      <div className={cn(
        "w-3 h-3 rounded-full border-2 border-white shadow-sm",
        config.dotColor
      )} />
    </div>
  );
};

export default ConnectionStatus;