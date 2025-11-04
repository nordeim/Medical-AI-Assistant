/**
 * UX Optimization Type Definitions
 * Comprehensive TypeScript interfaces for medical-grade UX optimization
 */

// Core optimization interfaces
export interface OptimizationConfig {
  enablePatientChatOptimization: boolean;
  enableNurseDashboardOptimization: boolean;
  enableResponsiveDesignOptimization: boolean;
  enableAccessibilityFeatures: boolean;
  enableUserOnboardingTraining: boolean;
  enableUserFeedbackCollection: boolean;
  enablePerformanceOptimization: boolean;
}

// Patient chat optimization types
export interface PatientChatProps {
  patientId: string;
  onEmergencyDetected: (severity: 'high' | 'critical', message: string) => void;
  accessibilitySettings?: AccessibilitySettings;
}

export interface ChatMessage {
  id: string;
  content: string;
  sender: 'patient' | 'ai';
  timestamp: Date;
  emergencyLevel?: 'none' | 'high' | 'critical';
}

export interface EmergencyKeyword {
  keyword: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  category: 'cardiac' | 'respiratory' | 'neurological' | 'trauma' | 'mental_health';
}

// Dashboard optimization types
export interface Patient {
  id: string;
  name: string;
  age: number;
  condition: string;
  priority: 'critical' | 'high' | 'medium' | 'low';
  waitTime: number;
  lastContact: Date;
  notes: string;
  vitals?: {
    heartRate: number;
    bloodPressure: string;
    temperature: number;
    oxygenSaturation: number;
  };
}

export interface DashboardLayout {
  widgets: {
    activePatients: boolean;
    criticalAlerts: boolean;
    recentMessages: boolean;
    quickActions: boolean;
    vitalsMonitor: boolean;
    schedule: boolean;
  };
  compactMode: boolean;
  autoRefresh: boolean;
  refreshInterval: number;
}

export interface QuickAction {
  id: string;
  label: string;
  icon: string;
  action: () => void;
  category: 'emergency' | 'communication' | 'documentation' | 'assessment';
  keyboardShortcut?: string;
}

// Responsive design types
export interface TouchGestures {
  onSwipeLeft?: () => void;
  onSwipeRight?: () => void;
  onSwipeUp?: () => void;
  onSwipeDown?: () => void;
  onPinchIn?: () => void;
  onPinchOut?: () => void;
}

export interface EMRIntegration {
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

export interface MobileOptimizations {
  dataCompression: boolean;
  offlineMode: boolean;
  batteryOptimization: boolean;
  touchFriendlyUI: boolean;
  voiceCommands: boolean;
  emergencyQuickActions: boolean;
}

export interface ResponsiveDesignProps {
  content: React.ReactNode;
  deviceType: 'desktop' | 'tablet' | 'mobile';
  emrIntegration?: EMRIntegration;
  mobileOptimizations?: Partial<MobileOptimizations>;
  touchGestures?: TouchGestures;
  onEmergencyAction?: (action: string) => void;
}

// Accessibility types
export interface AccessibilitySettings {
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

export interface ScreenReaderSupport {
  announcePageChanges: boolean;
  structuredNavigation: boolean;
  landmarkRoles: boolean;
  ariaLiveRegions: boolean;
  skipLinks: boolean;
}

export interface KeyboardNavigation {
  tabOrder: 'default' | 'custom' | 'smart';
  keyboardShortcuts: {
    [key: string]: () => void;
  };
  focusManagement: boolean;
  trapFocus: boolean;
}

export interface VisionAccessibility {
  highContrast: boolean;
  fontSize: 'small' | 'medium' | 'large' | 'extra-large';
  colorBlindSupport: boolean;
  screenReader: boolean;
  voiceCommands: boolean;
  textToSpeech: boolean;
}

export interface MotorAccessibility {
  voiceInput: boolean;
  switchNavigation: boolean;
  eyeTracking: boolean;
  footPedal: boolean;
  singleSwitch: boolean;
  gestureControl: boolean;
}

export interface CognitiveAccessibility {
  simplifiedInterface: boolean;
  memoryAids: boolean;
  stepByStep: boolean;
  timeExtensions: boolean;
  errorTolerance: boolean;
  consistentLayout: boolean;
}

// Training and onboarding types
export interface TrainingModule {
  id: string;
  title: string;
  description: string;
  category: 'safety' | 'workflow' | 'emergency' | 'documentation' | 'communication';
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  duration: number;
  prerequisites?: string[];
  content: TrainingContent[];
  assessment: Assessment;
}

export interface TrainingContent {
  type: 'text' | 'video' | 'interactive' | 'simulation' | 'quiz';
  title: string;
  content: string;
  mediaUrl?: string;
  interactiveElements?: any[];
}

export interface Assessment {
  questions: Question[];
  passingScore: number;
  maxAttempts: number;
}

export interface Question {
  id: string;
  type: 'multiple-choice' | 'true-false' | 'scenario' | 'simulation';
  question: string;
  options?: string[];
  correctAnswer: string | string[];
  explanation: string;
  category: 'safety' | 'emergency' | 'workflow' | 'documentation';
}

export interface UserProgress {
  userId: string;
  completedModules: string[];
  currentModule: string | null;
  moduleScores: { [moduleId: string]: number };
  attempts: { [moduleId: string]: number };
  lastAccess: Date;
  certifications: string[];
}

// Medical workflow types
export interface MedicalWorkflow {
  id: string;
  name: string;
  category: 'admission' | 'discharge' | 'emergency' | 'medication' | 'vitals';
  steps: WorkflowStep[];
  safetyChecks: SafetyCheck[];
  emergencyProcedures: EmergencyProcedure[];
}

export interface WorkflowStep {
  id: string;
  title: string;
  description: string;
  order: number;
  required: boolean;
  safetyCritical: boolean;
  estimatedTime: number;
  dependencies?: string[];
}

export interface SafetyCheck {
  id: string;
  checkType: 'verification' | 'confirmation' | 'double-check';
  description: string;
  verification: boolean;
  autoCheck?: boolean;
}

export interface EmergencyProcedure {
  trigger: string;
  steps: string[];
  escalation: string;
  notification: string[];
}

// Feedback collection types
export interface FeedbackSubmission {
  id: string;
  userId: string;
  userRole: 'nurse' | 'physician' | 'admin' | 'technician';
  category: 'usability' | 'clinical-effectiveness' | 'technical' | 'safety' | 'workflow';
  priority: 'low' | 'medium' | 'high' | 'critical';
  rating: {
    overall: number;
    usability: number;
    clinicalValue: number;
    technicalPerformance: number;
    workflowIntegration: number;
  };
  comments: {
    positive: string;
    negative: string;
    suggestions: string;
  };
  context: {
    deviceType: 'desktop' | 'tablet' | 'mobile';
    browserType: string;
    sessionDuration: number;
    tasksCompleted: number;
    errorCount: number;
  };
  timestamp: Date;
  status: 'new' | 'in-review' | 'addressed' | 'resolved';
  followUpRequired: boolean;
}

export interface ClinicalMetrics {
  responseTime: number;
  accuracyScore: number;
  workflowEfficiency: number;
  errorReduction: number;
  userSatisfaction: number;
  adoptionRate: number;
  taskCompletionRate: number;
}

export interface UsabilityMetrics {
  taskSuccessRate: number;
  timeOnTask: number;
  errorRate: number;
  userEngagement: number;
  learnabilityScore: number;
  memoryLoad: number;
  satisfactionScore: number;
}

export interface FeedbackAnalysis {
  totalSubmissions: number;
  averageRating: number;
  categoryDistribution: { [key: string]: number };
  priorityDistribution: { [key: string]: number };
  trendAnalysis: {
    weeklyVolume: number[];
    ratingTrends: number[];
    categoryTrends: { [key: string]: number[] };
  };
  topIssues: {
    issue: string;
    frequency: number;
    impact: 'low' | 'medium' | 'high';
    status: 'open' | 'in-progress' | 'resolved';
  }[];
  recommendations: string[];
  clinicalImprovements: {
    area: string;
    improvement: string;
    expectedImpact: string;
    effort: 'low' | 'medium' | 'high';
  }[];
}

// Performance optimization types
export interface PerformanceMetrics {
  responseTime: number;
  latency: number;
  throughput: number;
  errorRate: number;
  availability: number;
  dataTransferRate: number;
  memoryUsage: number;
  cpuUsage: number;
  networkQuality: 'excellent' | 'good' | 'fair' | 'poor' | 'offline';
  timestamp: Date;
}

export interface CacheConfig {
  enabled: boolean;
  maxSize: number;
  ttl: number;
  strategy: 'lru' | 'fifo' | 'lfu';
  criticalDataOnly: boolean;
}

export interface OfflineConfig {
  enabled: boolean;
  syncOnReconnect: boolean;
  queueSize: number;
  backgroundSync: boolean;
  criticalOperations: string[];
}

export interface PerformanceMonitoring {
  realTimeMetrics: boolean;
  alertingEnabled: boolean;
  performanceBudget: {
    responseTime: number;
    latency: number;
    memoryUsage: number;
  };
  thresholds: {
    warning: number;
    critical: number;
  };
}

export interface NetworkInfo {
  effectiveType: '4g' | '3g' | '2g' | 'slow-2g';
  downlink: number;
  rtt: number;
  saveData: boolean;
}

export interface StreamingConfig {
  enabled: boolean;
  batchSize: number;
  batchTimeout: number;
  priorityChannels: string[];
  quality: 'high' | 'balanced' | 'low';
  compression: boolean;
}

export interface QueueItem {
  id: string;
  type: 'patient_data' | 'vital_signs' | 'medication_update' | 'emergency_alert';
  data: any;
  timestamp: Date;
  priority: 'low' | 'normal' | 'high' | 'critical';
  attempts: number;
  maxAttempts: number;
}

// Utility types
export type DeviceType = 'mobile' | 'tablet' | 'desktop';
export type UserRole = 'nurse' | 'physician' | 'admin' | 'technician';
export type PriorityLevel = 'critical' | 'high' | 'medium' | 'low';
export type FeedbackCategory = 'usability' | 'clinical-effectiveness' | 'technical' | 'safety' | 'workflow';
export type TrainingCategory = 'safety' | 'workflow' | 'emergency' | 'documentation' | 'communication';
export type QuestionType = 'multiple-choice' | 'true-false' | 'scenario' | 'simulation';

// Constants
export const MEDICAL_CONSTANTS = {
  EMERGENCY_RESPONSE_TIME: 30,
  CRITICAL_ALERT_TIMEOUT: 10,
  DATA_SYNC_INTERVAL: 60,
  PERFORMANCE_MONITORING_INTERVAL: 5,
  CACHE_TTL: 300,
  OFFLINE_QUEUE_MAX_SIZE: 1000,
  TARGET_RESPONSE_TIME: 2000,
  TARGET_LATENCY: 500
} as const;

export const DEVICE_BREAKPOINTS = {
  mobile: 768,
  tablet: 1024,
  desktop: 1200
} as const;

export const PRIORITY_ORDER = {
  critical: 4,
  high: 3,
  medium: 2,
  low: 1
} as const;

export const ACCESSIBILITY_FONT_SIZES = ['small', 'medium', 'large', 'extra-large'] as const;