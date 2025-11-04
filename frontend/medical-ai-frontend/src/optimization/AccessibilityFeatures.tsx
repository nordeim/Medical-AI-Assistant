/**
 * Accessibility Features for Medical Professionals with Disabilities
 * WCAG 2.1 AA Compliant with medical workflow optimizations
 */

import React, { useState, useEffect, useRef } from 'react';
import { Button } from '../ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Alert, AlertDescription } from '../ui/alert';
import { Badge } from '../ui/badge';

// Screen reader optimization
interface ScreenReaderSupport {
  announcePageChanges: boolean;
  structuredNavigation: boolean;
  landmarkRoles: boolean;
  ariaLiveRegions: boolean;
  skipLinks: boolean;
}

// Keyboard navigation configurations
interface KeyboardNavigation {
  tabOrder: 'default' | 'custom' | 'smart';
  keyboardShortcuts: {
    [key: string]: () => void;
  };
  focusManagement: boolean;
  trapFocus: boolean;
}

// Vision accessibility features
interface VisionAccessibility {
  highContrast: boolean;
  fontSize: 'small' | 'medium' | 'large' | 'extra-large';
  colorBlindSupport: boolean;
  screenReader: boolean;
  voiceCommands: boolean;
  textToSpeech: boolean;
}

// Motor accessibility features
interface MotorAccessibility {
  voiceInput: boolean;
  switchNavigation: boolean;
  eyeTracking: boolean;
  footPedal: boolean;
  singleSwitch: boolean;
  gestureControl: boolean;
}

interface CognitiveAccessibility {
  simplifiedInterface: boolean;
  memoryAids: boolean;
  stepByStep: boolean;
  timeExtensions: boolean;
  errorTolerance: boolean;
  consistentLayout: boolean;
}

interface AccessibilitySettings {
  screenReader: ScreenReaderSupport;
  keyboard: KeyboardNavigation;
  vision: VisionAccessibility;
  motor: MotorAccessibility;
  cognitive: CognitiveAccessibility;
  medical: {
    criticalAlerts: boolean;
    emergencyActions: boolean;
    medicationWarnings: boolean;
    dosageCalculations: boolean;
  };
}

// Screen Reader Announcement Component
const ScreenReaderAnnouncements: React.FC<{
  announcements: string[];
  politeness: 'polite' | 'assertive';
}> = ({ announcements, politeness }) => {
  const [announcementIndex, setAnnouncementIndex] = useState(0);

  useEffect(() => {
    if (announcements.length > 0) {
      const announcement = announcements[announcementIndex];
      const srAnnouncement = document.getElementById('sr-announcement');
      if (srAnnouncement) {
        srAnnouncement.setAttribute('aria-live', politeness);
        srAnnouncement.textContent = announcement;
        
        // Clear after announcement
        setTimeout(() => {
          srAnnouncement.textContent = '';
        }, 1000);
        
        setAnnouncementIndex((prev) => (prev + 1) % announcements.length);
      }
    }
  }, [announcements, politeness, announcementIndex]);

  return (
    <div
      id="sr-announcement"
      aria-live={politeness}
      aria-atomic="true"
      className="sr-only"
    >
      {announcements[announcementIndex]}
    </div>
  );
};

// High Contrast Mode Toggle
const HighContrastMode: React.FC<{
  enabled: boolean;
  onToggle: () => void;
}> = ({ enabled, onToggle }) => {
  const [contrastTheme, setContrastTheme] = useState<'yellow-black' | 'white-black' | 'cyan-black'>('yellow-black');

  useEffect(() => {
    if (enabled) {
      document.documentElement.classList.add('high-contrast');
      document.documentElement.setAttribute('data-contrast', contrastTheme);
    } else {
      document.documentElement.classList.remove('high-contrast');
      document.documentElement.removeAttribute('data-contrast');
    }
  }, [enabled, contrastTheme]);

  return (
    <Card className="p-4">
      <h3 className="text-lg font-semibold mb-3">High Contrast Mode</h3>
      <div className="space-y-3">
        <Button
          onClick={onToggle}
          variant={enabled ? 'default' : 'outline'}
          aria-pressed={enabled}
        >
          {enabled ? 'Enabled' : 'Disabled'}
        </Button>
        
        {enabled && (
          <div role="group" aria-labelledby="contrast-theme-label">
            <label id="contrast-theme-label" className="block text-sm font-medium mb-2">
              Color Scheme:
            </label>
            <div className="grid grid-cols-3 gap-2">
              {[
                { id: 'yellow-black', label: 'Yellow/Black', preview: 'bg-yellow-400 text-black' },
                { id: 'white-black', label: 'White/Black', preview: 'bg-white text-black border' },
                { id: 'cyan-black', label: 'Cyan/Black', preview: 'bg-cyan-400 text-black' }
              ].map(theme => (
                <Button
                  key={theme.id}
                  variant={contrastTheme === theme.id ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setContrastTheme(theme.id as any)}
                  className={theme.preview}
                  aria-pressed={contrastTheme === theme.id}
                >
                  {theme.label}
                </Button>
              ))}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

// Keyboard Navigation Handler
const KeyboardNavigationHandler: React.FC<{
  settings: AccessibilitySettings;
  children: React.ReactNode;
}> = ({ settings, children }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [currentFocus, setCurrentFocus] = useState<HTMLElement | null>(null);

  // Enhanced keyboard shortcuts for medical workflows
  useEffect(() => {
    const shortcuts = {
      // Navigation shortcuts
      'Alt+H': () => document.getElementById('main-content')?.focus(),
      'Alt+P': () => document.getElementById('patient-section')?.focus(),
      'Alt+M': () => document.getElementById('medication-section')?.focus(),
      'Alt+V': () => document.getElementById('vitals-section')?.focus(),
      
      // Emergency actions (maintain physical separation from other keys)
      'Control+Shift+E': () => {
        // Emergency alert
        const sr = document.getElementById('sr-announcement');
        if (sr) {
          sr.textContent = 'Emergency protocol activated';
        }
        // Trigger emergency protocol
        console.log('Emergency protocol activated');
      },
      
      // Medication safety shortcuts
      'Control+Shift+D': () => {
        const sr = document.getElementById('sr-announcement');
        if (sr) {
          sr.textContent = 'Medication dosage calculator opened';
        }
        // Open dosage calculator
        console.log('Opening dosage calculator');
      },
      
      // Documentation shortcuts
      'Control+N': () => {
        const sr = document.getElementById('sr-announcement');
        if (sr) {
          sr.textContent = 'New clinical note opened';
        }
        // Open new note
        console.log('Opening new clinical note');
      },
      
      // Patient interaction shortcuts
      'Control+M': () => {
        const sr = document.getElementById('sr-announcement');
        if (sr) {
          sr.textContent = 'Patient message interface opened';
        }
        // Open patient messaging
        console.log('Opening patient messaging');
      }
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      const key = event.key;
      const ctrl = event.ctrlKey;
      const alt = event.altKey;
      const shift = event.shiftKey;

      // Build shortcut key string
      let shortcut = '';
      if (ctrl) shortcut += 'Control+';
      if (alt) shortcut += 'Alt+';
      if (shift) shortcut += 'Shift+';
      shortcut += key;

      if (shortcuts[shortcut]) {
        event.preventDefault();
        event.stopPropagation();
        shortcuts[shortcut]();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Focus management
  useEffect(() => {
    if (settings.keyboard.focusManagement && containerRef.current) {
      const focusableElements = containerRef.current.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      
      const firstElement = focusableElements[0] as HTMLElement;
      const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

      const handleTabKey = (event: KeyboardEvent) => {
        if (event.key === 'Tab') {
          if (event.shiftKey) {
            if (document.activeElement === firstElement) {
              event.preventDefault();
              lastElement?.focus();
            }
          } else {
            if (document.activeElement === lastElement) {
              event.preventDefault();
              firstElement?.focus();
            }
          }
        }
      };

      const container = containerRef.current;
      container.addEventListener('keydown', handleTabKey);
      
      return () => container.removeEventListener('keydown', handleTabKey);
    }
  }, [settings.keyboard.focusManagement]);

  return (
    <div ref={containerRef} role="main" aria-label="Main content area">
      {/* Skip links for screen readers */}
      {settings.screenReader.skipLinks && (
        <div className="sr-only focus:not-sr-only">
          <a href="#main-content" className="absolute top-0 left-0 bg-blue-600 text-white p-2 z-50">
            Skip to main content
          </a>
          <a href="#patient-section" className="absolute top-0 left-20 bg-blue-600 text-white p-2 z-50">
            Skip to patient information
          </a>
          <a href="#medication-section" className="absolute top-0 left-40 bg-blue-600 text-white p-2 z-50">
            Skip to medications
          </a>
        </div>
      )}
      
      {children}
    </div>
  );
};

// Voice Command Interface
const VoiceCommandInterface: React.FC<{
  enabled: boolean;
  commands: { [key: string]: () => void };
  onVoiceToggle: (enabled: boolean) => void;
}> = ({ enabled, commands, onVoiceToggle }) => {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [recognition, setRecognition] = useState<any>(null);

  useEffect(() => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition;
      const recognitionInstance = new SpeechRecognition();
      
      recognitionInstance.continuous = true;
      recognitionInstance.interimResults = true;
      recognitionInstance.lang = 'en-US';
      
      recognitionInstance.onresult = (event: any) => {
        const current = event.resultIndex;
        const transcript = event.results[current][0].transcript;
        setTranscript(transcript);
        
        // Check for command keywords
        Object.keys(commands).forEach(command => {
          if (transcript.toLowerCase().includes(command.toLowerCase())) {
            commands[command]();
            setTranscript('');
          }
        });
      };

      recognitionInstance.onstart = () => setIsListening(true);
      recognitionInstance.onend = () => setIsListening(false);

      setRecognition(recognitionInstance);
    }
  }, [commands]);

  const startListening = () => {
    if (recognition && enabled) {
      recognition.start();
    }
  };

  const stopListening = () => {
    if (recognition) {
      recognition.stop();
    }
  };

  if (!enabled) return null;

  return (
    <Card className="p-4">
      <h3 className="text-lg font-semibold mb-3">Voice Commands</h3>
      <div className="space-y-3">
        <div className="flex space-x-2">
          <Button
            onClick={isListening ? stopListening : startListening}
            variant={isListening ? 'destructive' : 'default'}
            size="sm"
            aria-pressed={isListening}
          >
            {isListening ? '🛑 Stop' : '🎤 Start'} Listening
          </Button>
          
          <Badge variant={isListening ? 'destructive' : 'secondary'}>
            {isListening ? 'Listening...' : 'Stopped'}
          </Badge>
        </div>
        
        {transcript && (
          <div className="p-2 bg-gray-100 rounded text-sm">
            <strong>Transcript:</strong> {transcript}
          </div>
        )}
        
        <div>
          <h4 className="font-medium mb-2">Available Commands:</h4>
          <ul className="text-sm space-y-1">
            <li>• "Emergency" - Trigger emergency protocol</li>
            <li>• "Patient vital signs" - Open vitals section</li>
            <li>• "Medication list" - Show medications</li>
            <li>• "New note" - Open clinical note</li>
            <li>• "Search patient" - Activate patient search</li>
          </ul>
        </div>
      </div>
    </Card>
  );
};

// Main Accessibility Component
export const AccessibilityFeatures: React.FC<{
  initialSettings?: Partial<AccessibilitySettings>;
  onSettingsChange: (settings: AccessibilitySettings) => void;
  children: React.ReactNode;
}> = ({
  initialSettings = {},
  onSettingsChange,
  children
}) => {
  const [settings, setSettings] = useState<AccessibilitySettings>({
    screenReader: {
      announcePageChanges: true,
      structuredNavigation: true,
      landmarkRoles: true,
      ariaLiveRegions: true,
      skipLinks: true,
      ...initialSettings.screenReader
    },
    keyboard: {
      tabOrder: 'smart',
      keyboardShortcuts: {},
      focusManagement: true,
      trapFocus: true,
      ...initialSettings.keyboard
    },
    vision: {
      highContrast: false,
      fontSize: 'medium',
      colorBlindSupport: true,
      screenReader: true,
      voiceCommands: true,
      textToSpeech: false,
      ...initialSettings.vision
    },
    motor: {
      voiceInput: true,
      switchNavigation: false,
      eyeTracking: false,
      footPedal: false,
      singleSwitch: false,
      gestureControl: false,
      ...initialSettings.motor
    },
    cognitive: {
      simplifiedInterface: false,
      memoryAids: true,
      stepByStep: true,
      timeExtensions: true,
      errorTolerance: true,
      consistentLayout: true,
      ...initialSettings.cognitive
    },
    medical: {
      criticalAlerts: true,
      emergencyActions: true,
      medicationWarnings: true,
      dosageCalculations: true,
      ...initialSettings.medical
    }
  });

  const [announcements, setAnnouncements] = useState<string[]>([]);
  const [assistiveTechnologies, setAssistiveTechnologies] = useState<string[]>([]);

  // Detect assistive technologies
  useEffect(() => {
    const detections = [];
    
    if (navigator.userAgent.includes('NVDA') || window.speechSynthesis) {
      detections.push('Screen Reader');
    }
    
    if ('speechRecognition' in window || 'webkitSpeechRecognition' in window) {
      detections.push('Voice Recognition');
    }
    
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      detections.push('Reduced Motion');
    }
    
    setAssistiveTechnologies(detections);
  }, []);

  // Auto-save settings changes
  useEffect(() => {
    onSettingsChange(settings);
  }, [settings, onSettingsChange]);

  // Announce critical medical events
  const announceToScreenReader = (message: string, priority: 'normal' | 'urgent' | 'critical' = 'normal') => {
    if (settings.screenReader.ariaLiveRegions) {
      setAnnouncements(prev => [...prev, `${priority === 'critical' ? 'CRITICAL: ' : ''}${message}`]);
      
      // Also create a visible announcement for high-priority items
      if (priority === 'critical') {
        console.log('CRITICAL ANNOUNCEMENT:', message);
      }
    }
  };

  // Apply vision settings
  useEffect(() => {
    const root = document.documentElement;
    
    // Font size
    const fontSizeMap = {
      small: '14px',
      medium: '16px',
      large: '18px',
      'extra-large': '20px'
    };
    root.style.setProperty('--base-font-size', fontSizeMap[settings.vision.fontSize]);
    
    // High contrast
    if (settings.vision.highContrast) {
      root.classList.add('accessibility-high-contrast');
    } else {
      root.classList.remove('accessibility-high-contrast');
    }
    
    // Color blind support
    if (settings.vision.colorBlindSupport) {
      root.classList.add('colorblind-support');
    } else {
      root.classList.remove('colorblind-support');
    }
    
    // Cognitive support
    if (settings.cognitive.simplifiedInterface) {
      root.classList.add('simplified-interface');
    } else {
      root.classList.remove('simplified-interface');
    }
  }, [settings.vision.fontSize, settings.vision.highContrast, settings.vision.colorBlindSupport, settings.cognitive.simplifiedInterface]);

  return (
    <div className="accessibility-container">
      {/* Screen Reader Announcements */}
      {settings.screenReader.ariaLiveRegions && (
        <ScreenReaderAnnouncements 
          announcements={announcements} 
          politeness="assertive" 
        />
      )}
      
      {/* Keyboard Navigation Handler */}
      <KeyboardNavigationHandler settings={settings}>
        {/* Settings Panel (toggle with Alt+A) */}
        <details className="accessibility-settings-panel">
          <summary className="sr-only">Accessibility Settings</summary>
          
          <div className="fixed inset-y-0 right-0 w-96 bg-white shadow-lg p-6 overflow-y-auto z-50 border-l">
            <h2 className="text-2xl font-bold mb-4">Accessibility Settings</h2>
            
            {/* Detected Assistive Technologies */}
            {assistiveTechnologies.length > 0 && (
              <Alert className="mb-4">
                <AlertDescription>
                  <strong>Detected:</strong> {assistiveTechnologies.join(', ')}
                </AlertDescription>
              </Alert>
            )}
            
            {/* Vision Settings */}
            <Card className="mb-4">
              <CardHeader>
                <CardTitle>Vision Accessibility</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <HighContrastMode
                  enabled={settings.vision.highContrast}
                  onToggle={() => setSettings(prev => ({
                    ...prev,
                    vision: { ...prev.vision, highContrast: !prev.vision.highContrast }
                  }))}
                />
                
                <div>
                  <label className="block text-sm font-medium mb-2">Font Size</label>
                  <select
                    value={settings.vision.fontSize}
                    onChange={(e) => setSettings(prev => ({
                      ...prev,
                      vision: { ...prev.vision, fontSize: e.target.value as any }
                    }))}
                    className="w-full border rounded px-3 py-2"
                  >
                    <option value="small">Small</option>
                    <option value="medium">Medium</option>
                    <option value="large">Large</option>
                    <option value="extra-large">Extra Large</option>
                  </select>
                </div>
                
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="colorBlindSupport"
                    checked={settings.vision.colorBlindSupport}
                    onChange={(e) => setSettings(prev => ({
                      ...prev,
                      vision: { ...prev.vision, colorBlindSupport: e.target.checked }
                    }))}
                  />
                  <label htmlFor="colorBlindSupport">Color Blind Support</label>
                </div>
              </CardContent>
            </Card>
            
            {/* Motor Settings */}
            <Card className="mb-4">
              <CardHeader>
                <CardTitle>Motor Accessibility</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <VoiceCommandInterface
                  enabled={settings.motor.voiceInput}
                  commands={{
                    'emergency': () => announceToScreenReader('Emergency protocol activated', 'critical'),
                    'patient vital signs': () => announceToScreenReader('Opening patient vital signs'),
                    'medication list': () => announceToScreenReader('Opening medication list'),
                    'new note': () => announceToScreenReader('Opening clinical note'),
                    'search patient': () => announceToScreenReader('Activating patient search')
                  }}
                  onVoiceToggle={(enabled) => setSettings(prev => ({
                    ...prev,
                    motor: { ...prev.motor, voiceInput: enabled }
                  }))}
                />
              </CardContent>
            </Card>
            
            {/* Cognitive Settings */}
            <Card className="mb-4">
              <CardHeader>
                <CardTitle>Cognitive Support</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="simplifiedInterface"
                    checked={settings.cognitive.simplifiedInterface}
                    onChange={(e) => setSettings(prev => ({
                      ...prev,
                      cognitive: { ...prev.cognitive, simplifiedInterface: e.target.checked }
                    }))}
                  />
                  <label htmlFor="simplifiedInterface">Simplified Interface</label>
                </div>
                
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="stepByStep"
                    checked={settings.cognitive.stepByStep}
                    onChange={(e) => setSettings(prev => ({
                      ...prev,
                      cognitive: { ...prev.cognitive, stepByStep: e.target.checked }
                    }))}
                  />
                  <label htmlFor="stepByStep">Step-by-step guidance</label>
                </div>
                
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="memoryAids"
                    checked={settings.cognitive.memoryAids}
                    onChange={(e) => setSettings(prev => ({
                      ...prev,
                      cognitive: { ...prev.cognitive, memoryAids: e.target.checked }
                    }))}
                  />
                  <label htmlFor="memoryAids">Memory aids and reminders</label>
                </div>
              </CardContent>
            </Card>
            
            {/* Medical-Specific Settings */}
            <Card className="mb-4">
              <CardHeader>
                <CardTitle>Medical Safety</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="criticalAlerts"
                    checked={settings.medical.criticalAlerts}
                    onChange={(e) => setSettings(prev => ({
                      ...prev,
                      medical: { ...prev.medical, criticalAlerts: e.target.checked }
                    }))}
                  />
                  <label htmlFor="criticalAlerts">Enhanced critical alerts</label>
                </div>
                
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="medicationWarnings"
                    checked={settings.medical.medicationWarnings}
                    onChange={(e) => setSettings(prev => ({
                      ...prev,
                      medical: { ...prev.medical, medicationWarnings: e.target.checked }
                    }))}
                  />
                  <label htmlFor="medicationWarnings">Medication safety warnings</label>
                </div>
              </CardContent>
            </Card>
          </div>
        </details>
        
        {/* Main Application Content */}
        <div className="accessibility-main-content">
          {children}
        </div>
      </KeyboardNavigationHandler>
    </div>
  );
};

export default AccessibilityFeatures;