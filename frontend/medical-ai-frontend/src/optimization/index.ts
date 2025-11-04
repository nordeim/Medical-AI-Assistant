/**
 * UX Optimization Module Index
 * Comprehensive medical-grade usability optimization
 */

// Core optimization components
export { default as PatientChatOptimization } from './PatientChatOptimization';
export { default as NurseDashboardOptimization } from './NurseDashboardOptimization';
export { default as ResponsiveDesignOptimization } from './ResponsiveDesignOptimization';
export { default as AccessibilityFeatures } from './AccessibilityFeatures';
export { default as UserOnboardingTraining } from './UserOnboardingTraining';
export { default as UserFeedbackCollection } from './UserFeedbackCollection';
export { default as PerformanceOptimization } from './PerformanceOptimization';

// Main integration component that combines all optimizations
export { MedicalAppOptimizationProvider, useOptimizationContext } from './MedicalAppOptimizationProvider';

// Type definitions
export type {
  AccessibilitySettings,
  PatientChatProps,
  DashboardLayout,
  ResponsiveDesignProps,
  UserProgress,
  FeedbackSubmission,
  PerformanceMetrics,
  OfflineConfig,
  CacheConfig
} from './types';

// Optimization configuration
export interface MedicalOptimizationConfig {
  // Feature toggles
  enablePatientChatOptimization: boolean;
  enableNurseDashboardOptimization: boolean;
  enableResponsiveDesignOptimization: boolean;
  enableAccessibilityFeatures: boolean;
  enableUserOnboardingTraining: boolean;
  enableUserFeedbackCollection: boolean;
  enablePerformanceOptimization: boolean;

  // Accessibility settings
  accessibility: {
    defaultFontSize: 'small' | 'medium' | 'large' | 'extra-large';
    highContrastDefault: boolean;
    screenReaderModeDefault: boolean;
    voiceInputEnabled: boolean;
    keyboardNavigationDefault: boolean;
  };

  // Performance settings
  performance: {
    targetResponseTime: number; // milliseconds
    targetLatency: number; // milliseconds
    enableCaching: boolean;
    cacheSizeLimit: number; // MB
    offlineModeEnabled: boolean;
    realTimeMonitoring: boolean;
  };

  // Medical workflow settings
  medicalWorkflow: {
    emergencyDetectionEnabled: boolean;
    criticalAlertThresholds: {
      responseTime: number;
      errorRate: number;
      availability: number;
    };
    trainingModulesEnabled: boolean;
    feedbackCollectionEnabled: boolean;
  };

  // Responsive design breakpoints
  responsive: {
    mobile: {
      maxWidth: number;
      enableTouchOptimization: boolean;
      enableVoiceCommands: boolean;
    };
    tablet: {
      minWidth: number;
      maxWidth: number;
      enableMultiColumn: boolean;
    };
    desktop: {
      minWidth: number;
      enableFullFeatureSet: boolean;
    };
  };
}

// Default configuration
export const defaultOptimizationConfig: MedicalOptimizationConfig = {
  enablePatientChatOptimization: true,
  enableNurseDashboardOptimization: true,
  enableResponsiveDesignOptimization: true,
  enableAccessibilityFeatures: true,
  enableUserOnboardingTraining: true,
  enableUserFeedbackCollection: true,
  enablePerformanceOptimization: true,

  accessibility: {
    defaultFontSize: 'medium',
    highContrastDefault: false,
    screenReaderModeDefault: false,
    voiceInputEnabled: true,
    keyboardNavigationDefault: true
  },

  performance: {
    targetResponseTime: 2000, // 2 seconds
    targetLatency: 500, // 500ms
    enableCaching: true,
    cacheSizeLimit: 50, // 50MB
    offlineModeEnabled: true,
    realTimeMonitoring: true
  },

  medicalWorkflow: {
    emergencyDetectionEnabled: true,
    criticalAlertThresholds: {
      responseTime: 3000,
      errorRate: 5,
      availability: 95
    },
    trainingModulesEnabled: true,
    feedbackCollectionEnabled: true
  },

  responsive: {
    mobile: {
      maxWidth: 768,
      enableTouchOptimization: true,
      enableVoiceCommands: true
    },
    tablet: {
      minWidth: 769,
      maxWidth: 1024,
      enableMultiColumn: true
    },
    desktop: {
      minWidth: 1025,
      enableFullFeatureSet: true
    }
  }
};

// Utility functions
export const isMobile = (): boolean => {
  return window.innerWidth <= 768;
};

export const isTablet = (): boolean => {
  return window.innerWidth > 768 && window.innerWidth <= 1024;
};

export const isDesktop = (): boolean => {
  return window.innerWidth > 1024;
};

export const getDeviceType = (): 'mobile' | 'tablet' | 'desktop' => {
  if (isMobile()) return 'mobile';
  if (isTablet()) return 'tablet';
  return 'desktop';
};

export const checkAccessibilitySupport = (): {
  screenReader: boolean;
  voiceInput: boolean;
  highContrast: boolean;
  reducedMotion: boolean;
} => {
  return {
    screenReader: navigator.userAgent.includes('NVDA') || 'speechSynthesis' in window,
    voiceInput: 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window,
    highContrast: window.matchMedia('(prefers-contrast: high)').matches,
    reducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches
  };
};

export const optimizeForMedicalContext = (config: Partial<MedicalOptimizationConfig> = {}): MedicalOptimizationConfig => {
  const deviceType = getDeviceType();
  const accessibilitySupport = checkAccessibilitySupport();
  
  return {
    ...defaultOptimizationConfig,
    ...config,
    
    // Device-specific optimizations
    responsive: {
      ...defaultOptimizationConfig.responsive,
      ...config.responsive,
      [deviceType]: {
        ...defaultOptimizationConfig.responsive[deviceType],
        ...config.responsive?.[deviceType]
      }
    },
    
    // Accessibility auto-detection
    accessibility: {
      ...defaultOptimizationConfig.accessibility,
      ...config.accessibility,
      // Auto-enable features based on detected support
      voiceInputEnabled: accessibilitySupport.voiceInput && config.accessibility?.voiceInputEnabled !== false,
      highContrastDefault: accessibilitySupport.highContrast || config.accessibility?.highContrastDefault === true
    }
  };
};

// Medical compliance helpers
export const validateMedicalCompliance = (config: MedicalOptimizationConfig): {
  isCompliant: boolean;
  issues: string[];
  recommendations: string[];
} => {
  const issues: string[] = [];
  const recommendations: string[] = [];

  // Check performance requirements
  if (config.performance.targetResponseTime > 2000) {
    issues.push('Response time target exceeds 2 second requirement');
    recommendations.push('Optimize API calls and implement caching to meet medical application standards');
  }

  if (config.performance.targetLatency > 500) {
    issues.push('Latency target exceeds 500ms requirement');
    recommendations.push('Consider edge computing or CDN for reduced latency');
  }

  // Check accessibility compliance
  if (!config.enableAccessibilityFeatures) {
    issues.push('Accessibility features are disabled');
    recommendations.push('Enable comprehensive accessibility features for medical professionals with disabilities');
  }

  // Check emergency detection
  if (!config.medicalWorkflow.emergencyDetectionEnabled) {
    issues.push('Emergency detection is disabled');
    recommendations.push('Enable emergency detection for patient safety');
  }

  // Check offline capabilities
  if (!config.performance.offlineModeEnabled) {
    issues.push('Offline mode is disabled');
    recommendations.push('Enable offline capabilities for reliable medical workflow continuity');
  }

  return {
    isCompliant: issues.length === 0,
    issues,
    recommendations
  };
};

// Integration helpers
export const createOptimizedComponent = <T extends React.ComponentType<any>>(
  Component: T,
  optimizations: {
    enableAccessibility?: boolean;
    enableResponsive?: boolean;
    enablePerformance?: boolean;
    enableMedicalFeatures?: boolean;
  } = {}
): T => {
  // This would typically be implemented with a higher-order component
  // that wraps the original component with optimization features
  return Component;
};

// Export constants for medical contexts
export const MEDICAL_CONSTANTS = {
  EMERGENCY_RESPONSE_TIME: 30, // seconds
  CRITICAL_ALERT_TIMEOUT: 10, // seconds
  DATA_SYNC_INTERVAL: 60, // seconds
  PERFORMANCE_MONITORING_INTERVAL: 5, // seconds
  CACHE_TTL: 300, // seconds
  OFFLINE_QUEUE_MAX_SIZE: 1000,
  ACCESSIBILITY_FONT_SIZES: ['small', 'medium', 'large', 'extra-large'] as const,
  DEVICE_BREAKPOINTS: {
    mobile: 768,
    tablet: 1024,
    desktop: 1200
  }
} as const;

// Export medical workflow priorities
export const MEDICAL_PRIORITIES = {
  CRITICAL: 'critical',
  HIGH: 'high',
  MEDIUM: 'medium',
  LOW: 'low'
} as const;

// Export medical workflow categories
export const MEDICAL_CATEGORIES = {
  SAFETY: 'safety',
  EMERGENCY: 'emergency',
  WORKFLOW: 'workflow',
  DOCUMENTATION: 'documentation',
  COMMUNICATION: 'communication'
} as const;