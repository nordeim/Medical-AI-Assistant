/**
 * Responsive Design Optimization for Mobile and Tablet Devices
 * Optimized for EMR integration and handheld usage
 */

import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { useMobile } from '../hooks/use-mobile';

// Touch gesture handling
interface TouchGestures {
  onSwipeLeft?: () => void;
  onSwipeRight?: () => void;
  onSwipeUp?: () => void;
  onSwipeDown?: () => void;
  onPinchIn?: () => void;
  onPinchOut?: () => void;
}

// EMR integration interface
interface EMRIntegration {
  patientData: {
    demographics: any;
    medicalHistory: any[];
    currentMedications: any[];
    allergies: any[];
  };
  vitalSigns: {
    timestamp: Date;
    values: {
      heartRate: number;
      bloodPressure: { systolic: number; diastolic: number };
      temperature: number;
      respiratoryRate: number;
      oxygenSaturation: number;
      weight: number;
      height: number;
    };
  }[];
  recentVisits: any[];
  labResults: any[];
}

// Mobile-specific optimizations
interface MobileOptimizations {
  dataCompression: boolean;
  offlineMode: boolean;
  batteryOptimization: boolean;
  touchFriendlyUI: boolean;
  voiceCommands: boolean;
  emergencyQuickActions: boolean;
}

interface ResponsiveDesignProps {
  content: React.ReactNode;
  deviceType: 'desktop' | 'tablet' | 'mobile';
  emrIntegration?: EMRIntegration;
  mobileOptimizations?: Partial<MobileOptimizations>;
  touchGestures?: TouchGestures;
  onEmergencyAction?: (action: string) => void;
}

// Touch gesture detection hook
const useTouchGestures = (elementRef: React.RefObject<HTMLElement>, gestures: TouchGestures) => {
  useEffect(() => {
    if (!elementRef.current) return;

    let startX = 0;
    let startY = 0;
    let startDistance = 0;

    const handleTouchStart = (e: TouchEvent) => {
      const touches = e.touches;
      
      if (touches.length === 1) {
        startX = touches[0].clientX;
        startY = touches[0].clientY;
      } else if (touches.length === 2) {
        startDistance = Math.sqrt(
          Math.pow(touches[1].clientX - touches[0].clientX, 2) +
          Math.pow(touches[1].clientY - touches[0].clientY, 2)
        );
      }
    };

    const handleTouchEnd = (e: TouchEvent) => {
      if (e.changedTouches.length === 1) {
        const endX = e.changedTouches[0].clientX;
        const endY = e.changedTouches[0].clientY;
        const deltaX = endX - startX;
        const deltaY = endY - startY;

        // Minimum swipe distance
        const minSwipeDistance = 50;

        if (Math.abs(deltaX) > Math.abs(deltaY)) {
          if (Math.abs(deltaX) > minSwipeDistance) {
            if (deltaX > 0 && gestures.onSwipeRight) {
              gestures.onSwipeRight();
            } else if (deltaX < 0 && gestures.onSwipeLeft) {
              gestures.onSwipeLeft();
            }
          }
        } else {
          if (Math.abs(deltaY) > minSwipeDistance) {
            if (deltaY > 0 && gestures.onSwipeDown) {
              gestures.onSwipeDown();
            } else if (deltaY < 0 && gestures.onSwipeUp) {
              gestures.onSwipeUp();
            }
          }
        }
      } else if (e.changedTouches.length === 2) {
        const touches = e.changedTouches;
        const endDistance = Math.sqrt(
          Math.pow(touches[1].clientX - touches[0].clientX, 2) +
          Math.pow(touches[1].clientY - touches[0].clientY, 2)
        );

        if (startDistance > 0) {
          const scale = endDistance / startDistance;
          if (scale > 1.1 && gestures.onPinchOut) {
            gestures.onPinchOut();
          } else if (scale < 0.9 && gestures.onPinchIn) {
            gestures.onPinchIn();
          }
        }
      }
    };

    const element = elementRef.current;
    element.addEventListener('touchstart', handleTouchStart, { passive: true });
    element.addEventListener('touchend', handleTouchEnd, { passive: true });

    return () => {
      element.removeEventListener('touchstart', handleTouchStart);
      element.removeEventListener('touchend', handleTouchEnd);
    };
  }, [elementRef, gestures]);
};

// Mobile EMR Dashboard Component
const MobileEMRDashboard: React.FC<{
  emrData: EMRIntegration;
  onVitalInput: (vitals: any) => void;
  onQuickNote: (note: string) => void;
}> = ({ emrData, onVitalInput, onQuickNote }) => {
  const [activeTab, setActiveTab] = useState('overview');
  const isMobile = useMobile();

  const latestVitals = emrData.vitalSigns[emrData.vitalSigns.length - 1]?.values;

  return (
    <div className="h-full flex flex-col">
      {/* Mobile Header with Emergency Actions */}
      <div className="bg-red-600 text-white p-3 flex items-center justify-between">
        <h1 className="font-bold text-lg">EMR Mobile</h1>
        <div className="flex space-x-2">
          <Button 
            variant="outline" 
            size="sm" 
            className="bg-white/20 border-white/30 text-white hover:bg-white/30"
            onClick={() => onQuickNote('Emergency note recorded')}
          >
            🚨 Alert
          </Button>
        </div>
      </div>

      {/* Tab Navigation for Mobile */}
      <div className="bg-white border-b">
        <div className="flex overflow-x-auto">
          {[
            { id: 'overview', label: 'Overview', icon: '📊' },
            { id: 'vitals', label: 'Vitals', icon: '💓' },
            { id: 'medications', label: 'Meds', icon: '💊' },
            { id: 'history', label: 'History', icon: '📋' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-shrink-0 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              <div className="flex items-center space-x-1">
                <span>{tab.icon}</span>
                <span className="hidden sm:inline">{tab.label}</span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-auto p-3">
        {activeTab === 'overview' && (
          <div className="space-y-3">
            {/* Critical Alerts */}
            {latestVitals && (
              <Card className="border-red-200">
                <CardContent className="p-4">
                  <h3 className="font-semibold text-red-800 mb-2">⚠️ Critical Alerts</h3>
                  <div className="space-y-1 text-sm">
                    {latestVitals.heartRate > 100 && <div className="text-red-600">• Tachycardia detected</div>}
                    {latestVitals.oxygenSaturation < 95 && <div className="text-red-600">• Low oxygen saturation</div>}
                    {latestVitals.temperature > 38 && <div className="text-red-600">• Fever detected</div>}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Quick Actions Grid */}
            <div className="grid grid-cols-2 gap-3">
              <Button
                variant="outline"
                className="h-20 flex flex-col items-center justify-center space-y-1"
                onClick={() => onVitalInput({ type: 'heartRate' })}
              >
                <span className="text-2xl">💓</span>
                <span className="text-xs">Heart Rate</span>
              </Button>
              
              <Button
                variant="outline"
                className="h-20 flex flex-col items-center justify-center space-y-1"
                onClick={() => onVitalInput({ type: 'bloodPressure' })}
              >
                <span className="text-2xl">🩺</span>
                <span className="text-xs">Blood Pressure</span>
              </Button>
              
              <Button
                variant="outline"
                className="h-20 flex flex-col items-center justify-center space-y-1"
                onClick={() => onVitalInput({ type: 'temperature' })}
              >
                <span className="text-2xl">🌡️</span>
                <span className="text-xs">Temperature</span>
              </Button>
              
              <Button
                variant="outline"
                className="h-20 flex flex-col items-center justify-center space-y-1"
                onClick={() => onVitalInput({ type: 'oxygenSaturation' })}
              >
                <span className="text-2xl">🫁</span>
                <span className="text-xs">O2 Saturation</span>
              </Button>
            </div>
          </div>
        )}

        {activeTab === 'vitals' && (
          <div className="space-y-3">
            <h3 className="font-semibold">Latest Vital Signs</h3>
            {emrData.vitalSigns.slice(-5).map((vital, index) => (
              <Card key={index}>
                <CardContent className="p-3">
                  <div className="text-sm text-gray-600 mb-2">
                    {new Date(vital.timestamp).toLocaleString()}
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>HR: {vital.values.heartRate} bpm</div>
                    <div>BP: {vital.values.bloodPressure.systolic}/{vital.values.bloodPressure.diastolic}</div>
                    <div>Temp: {vital.values.temperature}°C</div>
                    <div>O2: {vital.values.oxygenSaturation}%</div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        {activeTab === 'medications' && (
          <div className="space-y-3">
            <h3 className="font-semibold">Current Medications</h3>
            {emrData.patientData.currentMedications.map((med, index) => (
              <Card key={index}>
                <CardContent className="p-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-medium">{med.name}</div>
                      <div className="text-sm text-gray-600">{med.dosage} - {med.frequency}</div>
                    </div>
                    <Badge variant={med.priority === 'high' ? 'destructive' : 'secondary'}>
                      {med.priority}
                    </Badge>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        {activeTab === 'history' && (
          <div className="space-y-3">
            <h3 className="font-semibold">Recent Visits</h3>
            {emrData.recentVisits.map((visit, index) => (
              <Card key={index}>
                <CardContent className="p-3">
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="font-medium">{visit.reason}</div>
                      <div className="text-sm text-gray-600">{visit.provider}</div>
                      <div className="text-xs text-gray-500">{visit.date}</div>
                    </div>
                    <Badge variant="outline">{visit.status}</Badge>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

// Responsive Design Main Component
export const ResponsiveDesignOptimization: React.FC<ResponsiveDesignProps> = ({
  content,
  deviceType,
  emrIntegration,
  mobileOptimizations = {
    dataCompression: true,
    offlineMode: false,
    batteryOptimization: true,
    touchFriendlyUI: true,
    voiceCommands: true,
    emergencyQuickActions: true
  },
  touchGestures,
  onEmergencyAction
}) => {
  const contentRef = useRef<HTMLDivElement>(null);
  const [orientation, setOrientation] = useState<'portrait' | 'landscape'>('portrait');
  const [screenSize, setScreenSize] = useState({ width: 0, height: 0 });
  const isMobile = deviceType === 'mobile';
  const isTablet = deviceType === 'tablet';

  // Initialize touch gestures
  useTouchGestures(contentRef, touchGestures || {});

  // Screen size and orientation tracking
  useEffect(() => {
    const updateScreenSize = () => {
      setScreenSize({
        width: window.innerWidth,
        height: window.innerHeight
      });
      setOrientation(window.innerHeight > window.innerWidth ? 'portrait' : 'landscape');
    };

    updateScreenSize();
    window.addEventListener('resize', updateScreenSize);
    window.addEventListener('orientationchange', updateScreenSize);

    return () => {
      window.removeEventListener('resize', updateScreenSize);
      window.removeEventListener('orientationchange', updateScreenSize);
    };
  }, []);

  // Device-specific CSS classes
  const getDeviceClasses = () => {
    const baseClasses = 'w-full h-full';
    
    if (isMobile) {
      return `${baseClasses} max-w-sm mx-auto overflow-hidden`;
    } else if (isTablet) {
      return `${baseClasses} max-w-4xl mx-auto`;
    }
    
    return baseClasses;
  };

  // Optimized for mobile/tablet if we have EMR data
  if (emrIntegration && (isMobile || isTablet)) {
    return (
      <div className={getDeviceClasses()} ref={contentRef}>
        <MobileEMRDashboard
          emrData={emrIntegration}
          onVitalInput={(vitals) => {
            // Handle vital input
            console.log('Vital input:', vitals);
            onEmergencyAction?.('vital_recorded');
          }}
          onQuickNote={(note) => {
            // Handle quick note
            console.log('Quick note:', note);
            onEmergencyAction?.('note_added');
          }}
        />
      </div>
    );
  }

  return (
    <div className={getDeviceClasses()} ref={contentRef}>
      <div className={`${isMobile ? 'p-2' : isTablet ? 'p-4' : 'p-6'} h-full`}>
        {/* Device indicator for development */}
        {process.env.NODE_ENV === 'development' && (
          <div className="fixed top-2 right-2 z-50">
            <Badge variant="outline" className="bg-white/80 backdrop-blur">
              {deviceType} {orientation} {screenSize.width}x{screenSize.height}
            </Badge>
          </div>
        )}

        {/* Content with responsive adjustments */}
        <div className={`
          ${isMobile ? 'space-y-3' : isTablet ? 'space-y-4' : 'space-y-6'}
          h-full overflow-auto
        `}>
          {React.cloneElement(content as React.ReactElement, {
            className: isMobile ? 'text-sm' : isTablet ? 'text-base' : 'text-lg',
            touchOptimized: mobileOptimizations.touchFriendlyUI,
            compressed: mobileOptimizations.dataCompression
          })}
        </div>

        {/* Mobile-specific bottom navigation */}
        {isMobile && (
          <div className="fixed bottom-0 left-0 right-0 bg-white border-t">
            <div className="grid grid-cols-4 gap-1 p-2">
              <Button variant="ghost" size="sm" className="flex flex-col items-center space-y-1">
                <span className="text-lg">🏠</span>
                <span className="text-xs">Home</span>
              </Button>
              <Button variant="ghost" size="sm" className="flex flex-col items-center space-y-1">
                <span className="text-lg">👥</span>
                <span className="text-xs">Patients</span>
              </Button>
              <Button variant="ghost" size="sm" className="flex flex-col items-center space-y-1">
                <span className="text-lg">📊</span>
                <span className="text-xs">Charts</span>
              </Button>
              <Button 
                variant="ghost" 
                size="sm" 
                className="flex flex-col items-center space-y-1"
                onClick={() => onEmergencyAction?.('emergency_menu')}
              >
                <span className="text-lg">🚨</span>
                <span className="text-xs">Emergency</span>
              </Button>
            </div>
          </div>
        )}

        {/* Tablet-specific multi-column layout */}
        {isTablet && (
          <div className="mt-4 grid grid-cols-2 gap-4">
            <Card>
              <CardContent className="p-4">
                <h3 className="font-semibold mb-2">Quick Actions</h3>
                <div className="grid grid-cols-2 gap-2">
                  <Button size="sm">Patient List</Button>
                  <Button size="sm">New Note</Button>
                  <Button size="sm">Vitals</Button>
                  <Button size="sm">Schedule</Button>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4">
                <h3 className="font-semibold mb-2">Recent Activity</h3>
                <div className="space-y-2 text-sm">
                  <div>• Vitals recorded for Patient A</div>
                  <div>• Medication updated for Patient B</div>
                  <div>• Alert cleared for Patient C</div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
};

export default ResponsiveDesignOptimization;