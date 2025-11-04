import React from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { useChat } from '../../contexts/ChatContext';
import { MessageCircle, Shield, Clock, AlertTriangle } from 'lucide-react';

const WelcomeMessage: React.FC = () => {
  const { createSession } = useChat();

  const handleStartChat = async () => {
    await createSession();
  };

  return (
    <div className="flex flex-col items-center justify-center h-full p-6 text-center space-y-6">
      {/* Welcome Icon */}
      <div className="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center">
        <MessageCircle className="w-10 h-10 text-blue-600" />
      </div>

      {/* Welcome Text */}
      <div className="space-y-2">
        <h1 className="text-2xl font-bold text-gray-900">
          Welcome to Medical AI Assistant
        </h1>
        <p className="text-gray-600 max-w-md">
          I'm here to help you with initial health assessment and triage. 
          This conversation will help me understand your concerns and provide 
          appropriate guidance.
        </p>
      </div>

      {/* Features */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-2xl">
        <Card className="p-4 text-left">
          <div className="flex items-start space-x-3">
            <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center flex-shrink-0">
              <Shield className="w-4 h-4 text-green-600" />
            </div>
            <div>
              <h3 className="font-medium text-gray-900 mb-1">Safe & Secure</h3>
              <p className="text-sm text-gray-600">
                Your privacy is protected with enterprise-grade security and HIPAA compliance.
              </p>
            </div>
          </div>
        </Card>

        <Card className="p-4 text-left">
          <div className="flex items-start space-x-3">
            <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0">
              <Clock className="w-4 h-4 text-blue-600" />
            </div>
            <div>
              <h3 className="font-medium text-gray-900 mb-1">24/7 Available</h3>
              <p className="text-sm text-gray-600">
                Get initial assessment and guidance anytime, anywhere.
              </p>
            </div>
          </div>
        </Card>

        <Card className="p-4 text-left">
          <div className="flex items-start space-x-3">
            <div className="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center flex-shrink-0">
              <MessageCircle className="w-4 h-4 text-purple-600" />
            </div>
            <div>
              <h3 className="font-medium text-gray-900 mb-1">Intelligent Triage</h3>
              <p className="text-sm text-gray-600">
                AI-powered analysis based on medical guidelines and best practices.
              </p>
            </div>
          </div>
        </Card>

        <Card className="p-4 text-left">
          <div className="flex items-start space-x-3">
            <div className="w-8 h-8 bg-yellow-100 rounded-full flex items-center justify-center flex-shrink-0">
              <AlertTriangle className="w-4 h-4 text-yellow-600" />
            </div>
            <div>
              <h3 className="font-medium text-gray-900 mb-1">Emergency Detection</h3>
              <p className="text-sm text-gray-600">
                Automatically detects urgent symptoms and provides appropriate warnings.
              </p>
            </div>
          </div>
        </Card>
      </div>

      {/* Start Button */}
      <Button 
        onClick={handleStartChat}
        size="lg"
        className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3"
      >
        Start Health Assessment
      </Button>

      {/* Disclaimer */}
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 max-w-2xl">
        <div className="flex items-start space-x-3">
          <AlertTriangle className="w-5 h-5 text-yellow-600 mt-0.5 flex-shrink-0" />
          <div className="text-left">
            <h4 className="font-medium text-yellow-800 mb-1">Important Notice</h4>
            <p className="text-sm text-yellow-700">
              This AI assistant is for informational and triage purposes only. 
              For medical emergencies, please call <strong>911</strong> immediately. 
              This system does not provide medical diagnoses, medical advice, or treatment recommendations. 
              Always consult with a qualified healthcare professional for medical concerns.
            </p>
          </div>
        </div>
      </div>

      {/* How it works */}
      <Card className="p-6 bg-gray-50 border-0 max-w-2xl w-full">
        <h3 className="font-semibold text-gray-900 mb-3">How it works:</h3>
        <div className="space-y-2 text-sm text-gray-600 text-left">
          <div className="flex items-center space-x-3">
            <div className="w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-medium">1</div>
            <span>Answer questions about your symptoms and medical history</span>
          </div>
          <div className="flex items-center space-x-3">
            <div className="w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-medium">2</div>
            <span>AI analyzes your information using medical guidelines</span>
          </div>
          <div className="flex items-center space-x-3">
            <div className="w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-medium">3</div>
            <span>Receive initial assessment and appropriate guidance</span>
          </div>
          <div className="flex items-center space-x-3">
            <div className="w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-medium">4</div>
            <span>Nurse review for higher priority cases</span>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default WelcomeMessage;