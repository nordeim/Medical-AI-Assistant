/**
 * Patient Chat Interface Optimization
 * Implements medical-grade usability with accessibility and emergency detection
 */

import React, { useState, useEffect, useRef } from 'react';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Textarea } from '../ui/textarea';
import { Card } from '../ui/card';
import { Badge } from '../ui/badge';
import { Alert, AlertDescription } from '../ui/alert';
import { useToast } from '../hooks/use-toast';
import { ChatMessage } from '../types';

// Emergency keyword patterns for real-time detection
const EMERGENCY_KEYWORDS = [
  'chest pain', 'heart attack', 'can\'t breathe', 'difficulty breathing',
  'severe bleeding', 'loss of consciousness', 'stroke symptoms',
  'suicidal', 'overdose', 'severe injury', 'choking', 'seizure'
];

// Accessibility configurations
interface AccessibilitySettings {
  fontSize: 'small' | 'medium' | 'large' | 'extra-large';
  highContrast: boolean;
  screenReaderMode: boolean;
  voiceInputEnabled: boolean;
  simplifiedLanguage: boolean;
}

interface PatientChatProps {
  patientId: string;
  onEmergencyDetected: (severity: 'high' | 'critical', message: string) => void;
  accessibilitySettings?: AccessibilitySettings;
}

export const PatientChatOptimization: React.FC<PatientChatProps> = ({
  patientId,
  onEmergencyDetected,
  accessibilitySettings = {
    fontSize: 'medium',
    highContrast: false,
    screenReaderMode: false,
    voiceInputEnabled: true,
    simplifiedLanguage: false
  }
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [emergencyLevel, setEmergencyLevel] = useState<'none' | 'high' | 'critical'>('none');
  const [accessibilityMode, setAccessibilityMode] = useState<AccessibilitySettings>(accessibilitySettings);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const { toast } = useToast();

  // Emergency detection algorithm
  const detectEmergencyLevel = (text: string): 'none' | 'high' | 'critical' => {
    const lowerText = text.toLowerCase();
    let severityScore = 0;
    
    EMERGENCY_KEYWORDS.forEach(keyword => {
      if (lowerText.includes(keyword)) {
        severityScore += keyword.includes('heart attack') || keyword.includes('stroke') ? 3 : 1;
      }
    });

    if (severityScore >= 5) return 'critical';
    if (severityScore >= 2) return 'high';
    return 'none';
  };

  // Real-time message analysis
  useEffect(() => {
    if (inputText.trim()) {
      const severity = detectEmergencyLevel(inputText);
      setEmergencyLevel(severity);
      
      if (severity === 'critical') {
        toast({
          title: "🚨 Critical Alert",
          description: "Potential emergency detected. Please contact emergency services.",
          variant: "destructive"
        });
        onEmergencyDetected('critical', inputText);
      } else if (severity === 'high') {
        onEmergencyDetected('high', inputText);
      }
    }
  }, [inputText, onEmergencyDetected, toast]);

  // Auto-scroll to latest message
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Keyboard shortcuts for accessibility
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Ctrl+Enter to send message
      if (event.ctrlKey && event.key === 'Enter') {
        handleSendMessage();
      }
      // Escape to clear input
      if (event.key === 'Escape') {
        setInputText('');
        inputRef.current?.focus();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleSendMessage = () => {
    if (!inputText.trim()) return;

    const newMessage: ChatMessage = {
      id: Date.now().toString(),
      content: inputText,
      sender: 'patient',
      timestamp: new Date(),
      emergencyLevel
    };

    setMessages(prev => [...prev, newMessage]);
    setInputText('');
    setIsTyping(true);

    // Simulate AI response
    setTimeout(() => {
      const response: ChatMessage = {
        id: (Date.now() + 1).toString(),
        content: generateAIResponse(inputText, accessibilityMode.simplifiedLanguage),
        sender: 'ai',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, response]);
      setIsTyping(false);
    }, 1500);
  };

  const generateAIResponse = (input: string, simplified: boolean): string => {
    // Simplified language mode for better comprehension
    if (simplified) {
      return "I understand your concern. Let me help you with this health issue. Please describe your symptoms in more detail so I can assist you better.";
    }
    
    return "Thank you for your message. Based on your symptoms, I recommend consulting with your healthcare provider. Please describe any additional symptoms or concerns you may have.";
  };

  // Dynamic font sizing based on accessibility settings
  const getFontSizeClass = () => {
    switch (accessibilityMode.fontSize) {
      case 'small': return 'text-sm';
      case 'medium': return 'text-base';
      case 'large': return 'text-lg';
      case 'extra-large': return 'text-xl';
      default: return 'text-base';
    }
  };

  const getButtonSize = () => {
    switch (accessibilityMode.fontSize) {
      case 'small': return 'sm';
      case 'medium': return 'md';
      case 'large': return 'lg';
      case 'extra-large': return 'lg';
      default: return 'md';
    }
  };

  return (
    <div className={`flex flex-col h-full ${accessibilityMode.highContrast ? 'bg-black text-white' : 'bg-white text-gray-900'}`}>
      {/* Chat Header with Emergency Status */}
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center space-x-2">
          <h2 className={`font-semibold ${getFontSizeClass()}`}>Patient Chat</h2>
          <Badge variant={emergencyLevel === 'critical' ? 'destructive' : emergencyLevel === 'high' ? 'secondary' : 'default'}>
            {emergencyLevel === 'critical' ? '🚨 Critical' : emergencyLevel === 'high' ? '⚠️ High Priority' : '✅ Normal'}
          </Badge>
        </div>
        
        {/* Emergency indicator */}
        {emergencyLevel === 'critical' && (
          <Alert className="w-auto">
            <AlertDescription className="text-red-600 font-bold">
              Contact emergency services immediately!
            </AlertDescription>
          </Alert>
        )}
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4" role="log" aria-label="Chat messages">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.sender === 'patient' ? 'justify-end' : 'justify-start'}`}
            aria-label={`${message.sender} message`}
          >
            <div
              className={`max-w-[80%] p-3 rounded-lg ${
                message.sender === 'patient'
                  ? 'bg-blue-500 text-white'
                  : accessibilityMode.highContrast
                  ? 'bg-gray-800 text-white border'
                  : 'bg-gray-200 text-gray-900'
              } ${getFontSizeClass()}`}
              role="article"
            >
              <p className="whitespace-pre-wrap">{message.content}</p>
              <div className="flex items-center justify-between mt-1 text-xs opacity-70">
                <span>{message.timestamp.toLocaleTimeString()}</span>
                {message.emergencyLevel !== 'none' && (
                  <Badge variant="destructive" className="text-xs">
                    {message.emergencyLevel === 'critical' ? 'Critical' : 'High'} Priority
                  </Badge>
                )}
              </div>
            </div>
          </div>
        ))}
        
        {/* Typing indicator */}
        {isTyping && (
          <div className="flex justify-start">
            <div className={`p-3 rounded-lg ${accessibilityMode.highContrast ? 'bg-gray-800' : 'bg-gray-200'}`}>
              <div className="flex space-x-1" aria-label="Assistant is typing">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t p-4">
        <div className="flex items-end space-x-2">
          <div className="flex-1">
            <Textarea
              ref={inputRef}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder={accessibilityMode.simplifiedLanguage 
                ? "Tell me about your health concern..." 
                : "Describe your symptoms or health concerns..."
              }
              className={`min-h-[80px] resize-none ${getFontSizeClass()}`}
              aria-label="Message input"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
            />
            {emergencyLevel !== 'none' && (
              <div className="mt-2">
                <Badge variant={emergencyLevel === 'critical' ? 'destructive' : 'secondary'}>
                  {emergencyLevel === 'critical' ? '🚨 Critical Alert' : '⚠️ High Priority'}
                </Badge>
              </div>
            )}
          </div>
          
          <div className="flex flex-col space-y-2">
            {/* Send button */}
            <Button
              onClick={handleSendMessage}
              disabled={!inputText.trim()}
              size={getButtonSize()}
              className="px-6"
              aria-label="Send message"
            >
              Send
            </Button>
            
            {/* Voice input button */}
            {accessibilityMode.voiceInputEnabled && (
              <Button
                variant="outline"
                size={getButtonSize()}
                onClick={() => {
                  // Voice input implementation would go here
                  toast({
                    title: "Voice Input",
                    description: "Voice input feature is not yet implemented"
                  });
                }}
                aria-label="Use voice input"
              >
                🎤
              </Button>
            )}
          </div>
        </div>
        
        {/* Accessibility controls */}
        <div className="mt-4 flex items-center space-x-4">
          <label className={`text-sm ${getFontSizeClass()}`}>Font Size:</label>
          <select
            value={accessibilityMode.fontSize}
            onChange={(e) => setAccessibilityMode(prev => ({ ...prev, fontSize: e.target.value as any }))}
            className={`border rounded px-2 py-1 ${getFontSizeClass()}`}
            aria-label="Select font size"
          >
            <option value="small">Small</option>
            <option value="medium">Medium</option>
            <option value="large">Large</option>
            <option value="extra-large">Extra Large</option>
          </select>
          
          <Button
            variant={accessibilityMode.highContrast ? "default" : "outline"}
            size="sm"
            onClick={() => setAccessibilityMode(prev => ({ ...prev, highContrast: !prev.highContrast }))}
            aria-label="Toggle high contrast mode"
          >
            High Contrast
          </Button>
          
          <Button
            variant={accessibilityMode.simplifiedLanguage ? "default" : "outline"}
            size="sm"
            onClick={() => setAccessibilityMode(prev => ({ ...prev, simplifiedLanguage: !prev.simplifiedLanguage }))}
            aria-label="Toggle simplified language mode"
          >
            Simplified Language
          </Button>
        </div>
      </div>
    </div>
  );
};

export default PatientChatOptimization;