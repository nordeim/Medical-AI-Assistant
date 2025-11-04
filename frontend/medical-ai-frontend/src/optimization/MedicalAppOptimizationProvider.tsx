/**
 * Medical Application Optimization Provider
 * Context provider for managing optimization state across the application
 */

import React, { createContext, useContext, useReducer, useEffect, ReactNode } from 'react';
import { MedicalOptimizationConfig, defaultOptimizationConfig } from './index';

// Optimization state interface
interface OptimizationState {
  config: MedicalOptimizationConfig;
  userPreferences: {
    accessibility: {
      fontSize: 'small' | 'medium' | 'large' | 'extra-large';
      highContrast: boolean;
      screenReaderMode: boolean;
      voiceInputEnabled: boolean;
      keyboardNavigationEnabled: boolean;
    };
    performance: {
      cacheEnabled: boolean;
      offlineModeEnabled: boolean;
      realTimeMonitoringEnabled: boolean;
    };
    workflow: {
      emergencyDetectionEnabled: boolean;
      autoSaveEnabled: boolean;
      quickActionsEnabled: boolean;
    };
  };
  sessionMetrics: {
    startTime: Date;
    taskCount: number;
    errorCount: number;
    responseTimeSum: number;
    lastUpdate: Date;
  };
  deviceInfo: {
    type: 'mobile' | 'tablet' | 'desktop';
    screenSize: { width: number; height: number };
    touchEnabled: boolean;
    voiceSupported: boolean;
  };
  medicalContext: {
    currentPatientId?: string;
    currentUserRole: 'nurse' | 'physician' | 'admin' | 'technician';
    emergencyMode: boolean;
    criticalAlertsActive: number;
  };
}

// Action types
type OptimizationAction =
  | { type: 'UPDATE_CONFIG'; payload: Partial<MedicalOptimizationConfig> }
  | { type: 'UPDATE_USER_PREFERENCES'; payload: Partial<OptimizationState['userPreferences']> }
  | { type: 'UPDATE_SESSION_METRICS'; payload: Partial<OptimizationState['sessionMetrics']> }
  | { type: 'UPDATE_DEVICE_INFO'; payload: Partial<OptimizationState['deviceInfo']> }
  | { type: 'UPDATE_MEDICAL_CONTEXT'; payload: Partial<OptimizationState['medicalContext']> }
  | { type: 'EMERGENCY_MODE'; payload: { enabled: boolean } }
  | { type: 'CRITICAL_ALERT'; payload: { action: 'increment' | 'decrement' } }
  | { type: 'TASK_COMPLETED'; payload: { responseTime: number; hadError?: boolean } }
  | { type: 'LOAD_PREFERENCES'; payload: Partial<OptimizationState> }
  | { type: 'RESET_STATE' };

// Initial state
const initialState: OptimizationState = {
  config: defaultOptimizationConfig,
  userPreferences: {
    accessibility: {
      fontSize: 'medium',
      highContrast: false,
      screenReaderMode: false,
      voiceInputEnabled: true,
      keyboardNavigationEnabled: true
    },
    performance: {
      cacheEnabled: true,
      offlineModeEnabled: true,
      realTimeMonitoringEnabled: true
    },
    workflow: {
      emergencyDetectionEnabled: true,
      autoSaveEnabled: true,
      quickActionsEnabled: true
    }
  },
  sessionMetrics: {
    startTime: new Date(),
    taskCount: 0,
    errorCount: 0,
    responseTimeSum: 0,
    lastUpdate: new Date()
  },
  deviceInfo: {
    type: 'desktop',
    screenSize: { width: window.innerWidth, height: window.innerHeight },
    touchEnabled: 'ontouchstart' in window,
    voiceSupported: 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window
  },
  medicalContext: {
    currentUserRole: 'nurse',
    emergencyMode: false,
    criticalAlertsActive: 0
  }
};

// Reducer function
const optimizationReducer = (state: OptimizationState, action: OptimizationAction): OptimizationState => {
  switch (action.type) {
    case 'UPDATE_CONFIG':
      return {
        ...state,
        config: { ...state.config, ...action.payload }
      };

    case 'UPDATE_USER_PREFERENCES':
      return {
        ...state,
        userPreferences: {
          ...state.userPreferences,
          ...action.payload,
          accessibility: { ...state.userPreferences.accessibility, ...action.payload.accessibility },
          performance: { ...state.userPreferences.performance, ...action.payload.performance },
          workflow: { ...state.userPreferences.workflow, ...action.payload.workflow }
        }
      };

    case 'UPDATE_SESSION_METRICS':
      return {
        ...state,
        sessionMetrics: {
          ...state.sessionMetrics,
          ...action.payload,
          lastUpdate: new Date()
        }
      };

    case 'UPDATE_DEVICE_INFO':
      return {
        ...state,
        deviceInfo: {
          ...state.deviceInfo,
          ...action.payload
        }
      };

    case 'UPDATE_MEDICAL_CONTEXT':
      return {
        ...state,
        medicalContext: {
          ...state.medicalContext,
          ...action.payload
        }
      };

    case 'EMERGENCY_MODE':
      return {
        ...state,
        medicalContext: {
          ...state.medicalContext,
          emergencyMode: action.payload.enabled
        }
      };

    case 'CRITICAL_ALERT':
      return {
        ...state,
        medicalContext: {
          ...state.medicalContext,
          criticalAlertsActive: Math.max(0, state.medicalContext.criticalAlertsActive + (action.payload.action === 'increment' ? 1 : -1))
        }
      };

    case 'TASK_COMPLETED':
      return {
        ...state,
        sessionMetrics: {
          ...state.sessionMetrics,
          taskCount: state.sessionMetrics.taskCount + 1,
          errorCount: state.sessionMetrics.errorCount + (action.payload.hadError ? 1 : 0),
          responseTimeSum: state.sessionMetrics.responseTimeSum + action.payload.responseTime,
          lastUpdate: new Date()
        }
      };

    case 'LOAD_PREFERENCES':
      return {
        ...state,
        ...action.payload,
        sessionMetrics: {
          ...state.sessionMetrics,
          startTime: new Date(), // Always reset start time on load
          lastUpdate: new Date()
        }
      };

    case 'RESET_STATE':
      return {
        ...initialState,
        sessionMetrics: {
          ...initialState.sessionMetrics,
          startTime: new Date()
        }
      };

    default:
      return state;
  }
};

// Context interface
interface OptimizationContextType {
  state: OptimizationState;
  dispatch: React.Dispatch<OptimizationAction>;
  
  // Helper functions
  updateConfig: (config: Partial<MedicalOptimizationConfig>) => void;
  updateAccessibilitySettings: (settings: Partial<OptimizationState['userPreferences']['accessibility']>) => void;
  updatePerformanceSettings: (settings: Partial<OptimizationState['userPreferences']['performance']>) => void;
  updateWorkflowSettings: (settings: Partial<OptimizationState['userPreferences']['workflow']>) => void;
  activateEmergencyMode: (enabled: boolean) => void;
  incrementCriticalAlerts: () => void;
  decrementCriticalAlerts: () => void;
  recordTaskCompletion: (responseTime: number, hadError?: boolean) => void;
  updateDeviceInfo: (info: Partial<OptimizationState['deviceInfo']>) => void;
  updateMedicalContext: (context: Partial<OptimizationState['medicalContext']>) => void;
  
  // Computed values
  getAverageResponseTime: () => number;
  getErrorRate: () => number;
  isOptimizationEnabled: (feature: keyof MedicalOptimizationConfig) => boolean;
  getDeviceType: () => 'mobile' | 'tablet' | 'desktop';
  isAccessibilityMode: () => boolean;
  isPerformanceMode: () => boolean;
  isEmergencyMode: () => boolean;
}

// Create context
const OptimizationContext = createContext<OptimizationContextType | undefined>(undefined);

// Provider component
interface MedicalAppOptimizationProviderProps {
  children: ReactNode;
  initialConfig?: Partial<MedicalOptimizationConfig>;
  initialUserRole?: 'nurse' | 'physician' | 'admin' | 'technician';
  autoSavePreferences?: boolean;
}

export const MedicalAppOptimizationProvider: React.FC<MedicalAppOptimizationProviderProps> = ({
  children,
  initialConfig = {},
  initialUserRole = 'nurse',
  autoSavePreferences = true
}) => {
  const [state, dispatch] = useReducer(optimizationReducer, {
    ...initialState,
    config: { ...defaultOptimizationConfig, ...initialConfig },
    medicalContext: {
      ...initialState.medicalContext,
      currentUserRole: initialUserRole
    }
  });

  // Auto-detect device capabilities
  useEffect(() => {
    const updateDeviceInfo = () => {
      const width = window.innerWidth;
      let deviceType: 'mobile' | 'tablet' | 'desktop';
      
      if (width <= 768) {
        deviceType = 'mobile';
      } else if (width <= 1024) {
        deviceType = 'tablet';
      } else {
        deviceType = 'desktop';
      }

      dispatch({
        type: 'UPDATE_DEVICE_INFO',
        payload: {
          type: deviceType,
          screenSize: { width, height: window.innerHeight },
          touchEnabled: 'ontouchstart' in window,
          voiceSupported: 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window
        }
      });
    };

    updateDeviceInfo();
    window.addEventListener('resize', updateDeviceInfo);
    return () => window.removeEventListener('resize', updateDeviceInfo);
  }, []);

  // Auto-save preferences to localStorage
  useEffect(() => {
    if (autoSavePreferences) {
      localStorage.setItem('medical-app-optimization-preferences', JSON.stringify({
        userPreferences: state.userPreferences,
        config: state.config,
        medicalContext: { currentUserRole: state.medicalContext.currentUserRole }
      }));
    }
  }, [state.userPreferences, state.config, state.medicalContext, autoSavePreferences]);

  // Load preferences from localStorage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem('medical-app-optimization-preferences');
      if (saved) {
        const parsed = JSON.parse(saved);
        dispatch({ type: 'LOAD_PREFERENCES', payload: parsed });
      }
    } catch (error) {
      console.warn('Failed to load optimization preferences:', error);
    }
  }, []);

  // Helper functions
  const updateConfig = (config: Partial<MedicalOptimizationConfig>) => {
    dispatch({ type: 'UPDATE_CONFIG', payload: config });
  };

  const updateAccessibilitySettings = (settings: Partial<OptimizationState['userPreferences']['accessibility']>) => {
    dispatch({
      type: 'UPDATE_USER_PREFERENCES',
      payload: { accessibility: settings }
    });
  };

  const updatePerformanceSettings = (settings: Partial<OptimizationState['userPreferences']['performance']>) => {
    dispatch({
      type: 'UPDATE_USER_PREFERENCES',
      payload: { performance: settings }
    });
  };

  const updateWorkflowSettings = (settings: Partial<OptimizationState['userPreferences']['workflow']>) => {
    dispatch({
      type: 'UPDATE_USER_PREFERENCES',
      payload: { workflow: settings }
    });
  };

  const activateEmergencyMode = (enabled: boolean) => {
    dispatch({ type: 'EMERGENCY_MODE', payload: { enabled } });
  };

  const incrementCriticalAlerts = () => {
    dispatch({ type: 'CRITICAL_ALERT', payload: { action: 'increment' } });
  };

  const decrementCriticalAlerts = () => {
    dispatch({ type: 'CRITICAL_ALERT', payload: { action: 'decrement' } });
  };

  const recordTaskCompletion = (responseTime: number, hadError?: boolean) => {
    dispatch({
      type: 'TASK_COMPLETED',
      payload: { responseTime, hadError }
    });
  };

  const updateDeviceInfo = (info: Partial<OptimizationState['deviceInfo']>) => {
    dispatch({ type: 'UPDATE_DEVICE_INFO', payload: info });
  };

  const updateMedicalContext = (context: Partial<OptimizationState['medicalContext']>) => {
    dispatch({ type: 'UPDATE_MEDICAL_CONTEXT', payload: context });
  };

  // Computed values
  const getAverageResponseTime = (): number => {
    if (state.sessionMetrics.taskCount === 0) return 0;
    return state.sessionMetrics.responseTimeSum / state.sessionMetrics.taskCount;
  };

  const getErrorRate = (): number => {
    if (state.sessionMetrics.taskCount === 0) return 0;
    return (state.sessionMetrics.errorCount / state.sessionMetrics.taskCount) * 100;
  };

  const isOptimizationEnabled = (feature: keyof MedicalOptimizationConfig): boolean => {
    return Boolean(state.config[feature]);
  };

  const getDeviceType = (): 'mobile' | 'tablet' | 'desktop' => {
    return state.deviceInfo.type;
  };

  const isAccessibilityMode = (): boolean => {
    return state.config.enableAccessibilityFeatures &&
           (state.userPreferences.accessibility.highContrast ||
            state.userPreferences.accessibility.screenReaderMode ||
            state.userPreferences.accessibility.fontSize !== 'medium');
  };

  const isPerformanceMode = (): boolean => {
    return state.config.enablePerformanceOptimization &&
           (state.userPreferences.performance.cacheEnabled ||
            state.userPreferences.performance.offlineModeEnabled);
  };

  const isEmergencyMode = (): boolean => {
    return state.medicalContext.emergencyMode || state.medicalContext.criticalAlertsActive > 0;
  };

  const contextValue: OptimizationContextType = {
    state,
    dispatch,
    updateConfig,
    updateAccessibilitySettings,
    updatePerformanceSettings,
    updateWorkflowSettings,
    activateEmergencyMode,
    incrementCriticalAlerts,
    decrementCriticalAlerts,
    recordTaskCompletion,
    updateDeviceInfo,
    updateMedicalContext,
    getAverageResponseTime,
    getErrorRate,
    isOptimizationEnabled,
    getDeviceType,
    isAccessibilityMode,
    isPerformanceMode,
    isEmergencyMode
  };

  return (
    <OptimizationContext.Provider value={contextValue}>
      {children}
    </OptimizationContext.Provider>
  );
};

// Hook to use optimization context
export const useOptimizationContext = (): OptimizationContextType => {
  const context = useContext(OptimizationContext);
  if (context === undefined) {
    throw new Error('useOptimizationContext must be used within a MedicalAppOptimizationProvider');
  }
  return context;
};

// Higher-order component for auto-optimization
export const withOptimization = <P extends object>(
  Component: React.ComponentType<P>,
  optimizations?: {
    enableAccessibility?: boolean;
    enablePerformance?: boolean;
    enableMedicalWorkflow?: boolean;
    enableResponsiveDesign?: boolean;
  }
) => {
  return (props: P) => {
    const optimizationContext = useOptimizationContext();
    
    return <Component {...props} optimizationContext={optimizationContext} />;
  };
};

export default MedicalAppOptimizationProvider;