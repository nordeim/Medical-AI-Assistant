import React from 'react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { AlertTriangle, X, RefreshCw } from 'lucide-react';

interface ErrorAlertProps {
  title?: string;
  message: string;
  onRetry?: () => void;
  onDismiss?: () => void;
  className?: string;
}

const ErrorAlert: React.FC<ErrorAlertProps> = ({
  title = 'Error',
  message,
  onRetry,
  onDismiss,
  className = ''
}) => {
  return (
    <Alert className={`border-red-200 bg-red-50 ${className}`}>
      <div className="flex items-start space-x-3">
        <AlertTriangle className="w-5 h-5 text-red-600 mt-0.5" />
        <div className="flex-1">
          <h3 className="font-medium text-red-900 mb-1">{title}</h3>
          <AlertDescription className="text-red-700">
            {message}
          </AlertDescription>
          
          {/* Error Actions */}
          <div className="mt-3 flex items-center space-x-2">
            {onRetry && (
              <Button
                variant="outline"
                size="sm"
                onClick={onRetry}
                className="border-red-300 text-red-700 hover:bg-red-100"
              >
                <RefreshCw className="w-4 h-4 mr-1" />
                Try Again
              </Button>
            )}
            
            {onDismiss && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onDismiss}
                className="text-red-700 hover:bg-red-100"
              >
                <X className="w-4 h-4 mr-1" />
                Dismiss
              </Button>
            )}
          </div>
        </div>
      </div>
    </Alert>
  );
};

export default ErrorAlert;