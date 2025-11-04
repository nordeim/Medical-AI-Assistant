import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useChat } from '../../contexts/ChatContext';
import { useAuth } from '../../contexts/AuthContext';
import { apiService } from '../../services/api';
import { Shield, FileText, CheckCircle, AlertTriangle } from 'lucide-react';
import { cn } from '@/lib/utils';

const ConsentModal: React.FC = () => {
  const { consentInfo, consentRequired, handleConsent } = useChat();
  const { user } = useAuth();
  const [responses, setResponses] = useState<Record<string, any>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (consentRequired && !consentInfo) {
      // Load consent information
      loadConsentInfo();
    }
  }, [consentRequired, consentInfo]);

  const loadConsentInfo = async () => {
    try {
      const response = await apiService.getActiveConsent();
      if (response.success && response.data) {
        // Handle consent info loading
        console.log('Consent info loaded:', response.data);
      }
    } catch (error) {
      console.error('Failed to load consent info:', error);
      setError('Failed to load consent information');
    }
  };

  const handleFieldChange = (fieldId: string, value: any) => {
    setResponses(prev => ({
      ...prev,
      [fieldId]: value
    }));
  };

  const handleSubmit = async () => {
    setIsSubmitting(true);
    setError(null);

    try {
      const success = await handleConsent(responses);
      if (!success) {
        setError('Failed to submit consent. Please try again.');
      }
    } catch (error) {
      setError('An error occurred while submitting consent.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const isFormValid = () => {
    if (!consentInfo?.required_fields) return true;
    
    return consentInfo.required_fields.every(fieldId => {
      const response = responses[fieldId];
      return response !== undefined && response !== '';
    });
  };

  if (!consentRequired) {
    return null;
  }

  // Default consent content if not loaded
  const defaultConsentContent = {
    title: "Medical AI Assistant Consent & Privacy Notice",
    content: `
This Medical AI Assistant is designed to provide initial health assessment and triage support. By using this service, you acknowledge and agree to the following:

**Purpose & Limitations:**
- This AI assistant provides informational guidance only and does not constitute medical advice, diagnosis, or treatment
- This system is not a substitute for professional medical care, diagnosis, or treatment
- For medical emergencies, always call 911 or go to your nearest emergency room

**Your Privacy & Data:**
- Your conversations are encrypted and stored securely
- Information is used only for providing assistance and improving our services
- We follow HIPAA-compliant practices to protect your health information
- Data may be reviewed by qualified medical professionals for quality assurance

**AI Limitations:**
- AI responses are based on general medical guidelines and may not be suitable for your specific situation
- The AI does not have access to your complete medical history
- Always verify AI recommendations with a healthcare professional

**Consent Requirements:**
- I understand this is for informational purposes only
- I will seek professional medical advice for serious concerns
- I consent to the secure storage and processing of my health information
- I am at least 18 years old or have parental/guardian consent

By continuing, you acknowledge that you have read, understood, and agree to these terms.
    `.trim(),
    required_fields: [
      'acknowledgment',
      'emergency_notice',
      'privacy_consent',
      'age_verification'
    ]
  };

  const consent = consentInfo || defaultConsentContent;

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <Card className="w-full max-w-4xl max-h-[90vh] overflow-hidden shadow-xl">
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="bg-blue-600 text-white p-6">
            <div className="flex items-center space-x-3">
              <Shield className="w-8 h-8" />
              <div>
                <h1 className="text-2xl font-bold">{consent.title}</h1>
                <p className="text-blue-100">Please review and accept to continue</p>
              </div>
            </div>
          </div>

          {/* Content */}
          <div className="flex-1 flex flex-col overflow-hidden">
            <ScrollArea className="flex-1 p-6">
              <div className="space-y-6">
                {/* Consent Content */}
                <div className="prose prose-sm max-w-none">
                  <div className="bg-gray-50 p-6 rounded-lg border">
                    <pre className="whitespace-pre-wrap font-sans text-sm text-gray-700 leading-relaxed">
                      {consent.content}
                    </pre>
                  </div>
                </div>

                {/* Consent Form */}
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-gray-900 flex items-center space-x-2">
                    <FileText className="w-5 h-5" />
                    <span>Required Acknowledgments</span>
                  </h3>

                  {/* Acknowledgment Checkboxes */}
                  <div className="space-y-4">
                    <div className="flex items-start space-x-3">
                      <Checkbox
                        id="acknowledgment"
                        checked={responses.acknowledgment || false}
                        onCheckedChange={(checked) => 
                          handleFieldChange('acknowledgment', checked)
                        }
                        className="mt-1"
                      />
                      <label 
                        htmlFor="acknowledgment" 
                        className="text-sm text-gray-700 cursor-pointer"
                      >
                        I understand that this AI assistant provides informational guidance only and is not a substitute for professional medical advice, diagnosis, or treatment.
                      </label>
                    </div>

                    <div className="flex items-start space-x-3">
                      <Checkbox
                        id="emergency_notice"
                        checked={responses.emergency_notice || false}
                        onCheckedChange={(checked) => 
                          handleFieldChange('emergency_notice', checked)
                        }
                        className="mt-1"
                      />
                      <label 
                        htmlFor="emergency_notice" 
                        className="text-sm text-gray-700 cursor-pointer"
                      >
                        I understand that for medical emergencies, I should call 911 or go to the nearest emergency room immediately.
                      </label>
                    </div>

                    <div className="flex items-start space-x-3">
                      <Checkbox
                        id="privacy_consent"
                        checked={responses.privacy_consent || false}
                        onCheckedChange={(checked) => 
                          handleFieldChange('privacy_consent', checked)
                        }
                        className="mt-1"
                      />
                      <label 
                        htmlFor="privacy_consent" 
                        className="text-sm text-gray-700 cursor-pointer"
                      >
                        I consent to the secure storage and processing of my health information for the purpose of providing medical AI assistance.
                      </label>
                    </div>

                    <div className="flex items-start space-x-3">
                      <Checkbox
                        id="age_verification"
                        checked={responses.age_verification || false}
                        onCheckedChange={(checked) => 
                          handleFieldChange('age_verification', checked)
                        }
                        className="mt-1"
                      />
                      <label 
                        htmlFor="age_verification" 
                        className="text-sm text-gray-700 cursor-pointer"
                      >
                        I confirm that I am at least 18 years old or have parental/guardian consent to use this service.
                      </label>
                    </div>
                  </div>
                </div>

                {/* Error Display */}
                {error && (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center space-x-2">
                    <AlertTriangle className="w-5 h-5 text-red-600" />
                    <p className="text-sm text-red-700">{error}</p>
                  </div>
                )}
              </div>
            </ScrollArea>

            {/* Footer */}
            <div className="border-t bg-gray-50 p-6">
              <div className="flex items-center justify-between">
                <div className="text-sm text-gray-600">
                  {user && (
                    <span>Signing as: {user.first_name || user.email}</span>
                  )}
                </div>
                
                <div className="flex items-center space-x-3">
                  <Button
                    variant="outline"
                    onClick={() => {
                      // Handle logout or redirect to home
                      window.location.href = '/';
                    }}
                  >
                    Decline & Exit
                  </Button>
                  
                  <Button
                    onClick={handleSubmit}
                    disabled={!isFormValid() || isSubmitting}
                    className="bg-blue-600 hover:bg-blue-700"
                  >
                    {isSubmitting ? (
                      <>
                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                        Submitting...
                      </>
                    ) : (
                      <>
                        <CheckCircle className="w-4 h-4 mr-2" />
                        I Accept & Continue
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default ConsentModal;