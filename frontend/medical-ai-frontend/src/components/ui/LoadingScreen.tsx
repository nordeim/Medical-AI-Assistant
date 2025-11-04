import React from 'react';
import { Card } from '@/components/ui/card';
import { Loader2 } from 'lucide-react';

interface LoadingScreenProps {
  message?: string;
  className?: string;
}

const LoadingScreen: React.FC<LoadingScreenProps> = ({ 
  message = 'Loading...', 
  className = '' 
}) => {
  return (
    <div className={`min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4 ${className}`}>
      <Card className="w-full max-w-md p-8 text-center">
        <div className="space-y-4">
          {/* Animated Logo */}
          <div className="w-16 h-16 mx-auto bg-blue-100 rounded-full flex items-center justify-center">
            <Loader2 className="w-8 h-8 text-blue-600 animate-spin" />
          </div>
          
          {/* Loading Message */}
          <div>
            <h2 className="text-lg font-semibold text-gray-900 mb-2">
              {message}
            </h2>
            <p className="text-sm text-gray-600">
              Please wait while we prepare your medical AI assistant...
            </p>
          </div>

          {/* Loading Animation */}
          <div className="flex justify-center space-x-1">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" />
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce delay-75" />
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce delay-150" />
          </div>
        </div>
      </Card>
    </div>
  );
};

export default LoadingScreen;