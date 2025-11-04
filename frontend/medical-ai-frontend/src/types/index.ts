// Core API and Chat Types
export interface User {
  id: string;
  email: string;
  role: 'patient' | 'nurse' | 'admin';
  first_name?: string;
  last_name?: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface Session {
  id: string;
  patient_id: string;
  status: 'active' | 'completed' | 'cancelled';
  created_at: string;
  ended_at?: string;
  metadata?: Record<string, any>;
}

export interface Message {
  id: string;
  session_id: string;
  sender_type: 'patient' | 'agent' | 'system';
  content: string;
  timestamp: string;
  metadata?: Record<string, any>;
  is_streaming?: boolean;
}

export interface ChatMessage extends Omit<Message, 'timestamp'> {
  sender: 'patient' | 'agent' | 'system';
  text: string;
  isUser: boolean;
  timestamp: Date;
  metadata?: {
    isStreaming?: boolean;
    tokenCount?: number;
    safetyFlagged?: boolean;
  };
}

export interface PatientAssessmentReport {
  id: string;
  session_id: string;
  chief_complaint: string;
  symptoms: string[];
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  urgency: 'routine' | 'urgent' | 'immediate';
  recommendations: string[];
  created_at: string;
  reviewed_by?: string;
  review_status?: 'pending' | 'approved' | 'override';
  nurse_notes?: string;
  has_red_flags?: boolean;
  red_flags?: string[];
  differential_diagnosis?: string[];
  assessment_data?: {
    history?: Record<string, any>;
    vital_signs?: Record<string, any>;
    symptoms_analysis?: Record<string, any>;
  };
  rag_sources?: RAGSource[];
  confidence_scores?: Record<string, number>;
  guideline_references?: GuidelineReference[];
}

export interface NurseQueueItem {
  id: string;
  session_id: string;
  patient_id: string;
  patient_name?: string;
  chief_complaint: string;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  urgency: 'routine' | 'urgent' | 'immediate';
  has_red_flags: boolean;
  created_at: string;
  priority_score?: number;
  status: 'pending' | 'in_review' | 'reviewed';
  wait_time_minutes?: number;
  assigned_nurse_id?: string;
}

export interface NurseQueueResponse {
  queue: NurseQueueItem[];
  total: number;
  urgent_count: number;
  immediate_count: number;
  red_flag_count: number;
}

export interface RAGSource {
  id: string;
  source_type: 'guideline' | 'knowledge_base' | 'protocol';
  title: string;
  content: string;
  relevance_score: number;
  reference_url?: string;
}

export interface GuidelineReference {
  guideline_id: string;
  title: string;
  section: string;
  recommendation: string;
  evidence_level?: string;
}

export interface NurseAction {
  action: 'approved' | 'override';
  notes?: string;
  override_reason?: string;
  recommended_actions?: string[];
}

export interface PARDetailView extends PatientAssessmentReport {
  session: Session;
  messages: Message[];
  patient: {
    id: string;
    name?: string;
    age?: number;
    gender?: string;
  };
}

export interface AnalyticsMetrics {
  total_pars_today: number;
  total_pars_week: number;
  avg_wait_time_minutes: number;
  avg_review_time_minutes: number;
  approval_rate: number;
  override_rate: number;
  red_flag_rate: number;
  high_risk_rate: number;
  pars_by_urgency: Record<string, number>;
  pars_by_hour: Array<{ hour: number; count: number }>;
  top_complaints: Array<{ complaint: string; count: number }>;
}

export interface ConsentInfo {
  id: string;
  version: string;
  title: string;
  content: string;
  required_fields: string[];
  is_active: boolean;
  created_at: string;
}

export interface ConsentResponse {
  id: string;
  consent_id: string;
  user_id: string;
  responses: Record<string, any>;
  signed_at: string;
  ip_address: string;
  user_agent: string;
}

export interface SafetyFilterLog {
  id: string;
  session_id: string;
  message_id?: string;
  filter_type: 'content' | 'red_flag' | 'safety';
  flagged_content: string;
  action_taken: 'blocked' | 'warned' | 'allowed';
  reason: string;
  timestamp: string;
}

export interface AuditLog {
  id: string;
  user_id: string;
  action: string;
  entity_type: string;
  entity_id: string;
  old_values?: Record<string, any>;
  new_values?: Record<string, any>;
  ip_address: string;
  user_agent: string;
  timestamp: string;
}

// WebSocket Types
export interface WebSocketMessage {
  type: 'message' | 'typing' | 'session_update' | 'session_join' | 'error' | 'consent_required' | 'red_flag_alert' | 'new_par' | 'par_update' | 'queue_update';
  payload: any;
  timestamp: string;
}

export interface ChatWebSocketMessage extends WebSocketMessage {
  type: 'message';
  payload: {
    message_id: string;
    session_id: string;
    sender_type: 'patient' | 'agent' | 'system';
    content: string;
    is_streaming?: boolean;
    metadata?: Record<string, any>;
  };
}

// API Response Types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: Record<string, any>;
  };
  pagination?: {
    page: number;
    per_page: number;
    total: number;
    total_pages: number;
  };
}

// UI State Types
export interface ChatState {
  session: Session | null;
  messages: Message[];
  isConnected: boolean;
  isConnecting: boolean;
  isTyping: boolean;
  isLoading: boolean;
  error: string | null;
  consentRequired: boolean;
  consentInfo: ConsentInfo | null;
  currentInput: string;
  isStreaming: boolean;
}

export interface AppState {
  user: User | null;
  isAuthenticated: boolean;
  currentSession: Session | null;
  chatState: ChatState;
  nurseQueue: NurseQueueItem[];
}

// Form Types
export interface LoginForm {
  email: string;
  password: string;
}

export interface RegistrationForm {
  email: string;
  password: string;
  confirmPassword: string;
  firstName?: string;
  lastName?: string;
}

export interface ConsentForm {
  [key: string]: any;
}

export interface ChatForm {
  message: string;
}

// Error Types
export interface AppError {
  code: string;
  message: string;
  details?: Record<string, any>;
  timestamp: string;
}

// Filter Types
export interface PARFilters {
  urgency?: ('routine' | 'urgent' | 'immediate')[];
  risk_level?: ('low' | 'medium' | 'high' | 'critical')[];
  has_red_flags?: boolean;
  date_from?: string;
  date_to?: string;
  search_query?: string;
  assigned_nurse?: string;
  status?: ('pending' | 'in_review' | 'reviewed')[];
}

export interface SortOption {
  field: 'created_at' | 'urgency' | 'risk_level' | 'wait_time';
  direction: 'asc' | 'desc';
}