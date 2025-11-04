import { 
  User, 
  Session, 
  Message, 
  PatientAssessmentReport, 
  NurseQueueItem, 
  ConsentInfo,
  ConsentResponse,
  ApiResponse,
  LoginForm,
  RegistrationForm,
  ChatForm
} from '../types';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

class ApiService {
  private baseURL: string;
  private token: string | null = null;

  constructor() {
    this.baseURL = API_BASE_URL;
    this.token = localStorage.getItem('authToken');
  }

  private async request<T>(
    endpoint: string, 
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseURL}${endpoint}`;
    
    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...(this.token && { Authorization: `Bearer ${this.token}` }),
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || 'API request failed');
      }

      return data;
    } catch (error) {
      console.error('API Request failed:', error);
      throw error;
    }
  }

  setToken(token: string) {
    this.token = token;
    localStorage.setItem('authToken', token);
  }

  clearToken() {
    this.token = null;
    localStorage.removeItem('authToken');
  }

  // Authentication endpoints
  async login(credentials: LoginForm): Promise<ApiResponse<{ user: User; access_token: string }>> {
    return this.request('/auth/login', {
      method: 'POST',
      body: JSON.stringify(credentials),
    });
  }

  async register(userData: RegistrationForm): Promise<ApiResponse<{ user: User; access_token: string }>> {
    return this.request('/auth/register', {
      method: 'POST',
      body: JSON.stringify(userData),
    });
  }

  async refreshToken(): Promise<ApiResponse<{ access_token: string }>> {
    return this.request('/auth/refresh', {
      method: 'POST',
    });
  }

  async getCurrentUser(): Promise<ApiResponse<User>> {
    return this.request('/auth/me');
  }

  // Session endpoints
  async createSession(patientId?: string): Promise<ApiResponse<Session>> {
    return this.request('/sessions', {
      method: 'POST',
      body: JSON.stringify({ patient_id: patientId }),
    });
  }

  async getSession(sessionId: string): Promise<ApiResponse<Session>> {
    return this.request(`/sessions/${sessionId}`);
  }

  async updateSessionStatus(sessionId: string, status: 'active' | 'completed' | 'cancelled'): Promise<ApiResponse<Session>> {
    return this.request(`/sessions/${sessionId}/status`, {
      method: 'PATCH',
      body: JSON.stringify({ status }),
    });
  }

  async getSessions(): Promise<ApiResponse<Session[]>> {
    return this.request('/sessions');
  }

  // Message endpoints
  async sendMessage(sessionId: string, content: string): Promise<ApiResponse<Message>> {
    return this.request(`/sessions/${sessionId}/messages`, {
      method: 'POST',
      body: JSON.stringify({ content }),
    });
  }

  async getSessionMessages(sessionId: string): Promise<ApiResponse<Message[]>> {
    return this.request(`/sessions/${sessionId}/messages`);
  }

  // PAR (Patient Assessment Report) endpoints
  async getPARs(): Promise<ApiResponse<PatientAssessmentReport[]>> {
    return this.request('/pars');
  }

  async getPAR(parId: string): Promise<ApiResponse<PatientAssessmentReport>> {
    return this.request(`/pars/${parId}`);
  }

  async reviewPAR(parId: string, review: { 
    status: 'approved' | 'override'; 
    notes?: string; 
  }): Promise<ApiResponse<PatientAssessmentReport>> {
    return this.request(`/pars/${parId}/review`, {
      method: 'PATCH',
      body: JSON.stringify(review),
    });
  }

  // Nurse queue endpoints
  async getNurseQueue(params?: {
    limit?: number;
    offset?: number;
    urgency?: string;
    has_red_flags?: boolean;
    risk_level?: string;
    status?: string;
  }): Promise<ApiResponse<NurseQueueItem[]>> {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          searchParams.append(key, value.toString());
        }
      });
    }
    const query = searchParams.toString();
    return this.request(`/pars/queue${query ? `?${query}` : ''}`);
  }

  // PAR Detail with full context
  async getPARDetail(parId: string): Promise<ApiResponse<PatientAssessmentReport>> {
    return this.request(`/pars/${parId}/detail`);
  }

  // Submit nurse action (approve/override)
  async submitNurseAction(
    parId: string,
    action: {
      action: 'approved' | 'override';
      notes?: string;
      override_reason?: string;
      recommended_actions?: string[];
    }
  ): Promise<ApiResponse<PatientAssessmentReport>> {
    return this.request(`/pars/${parId}/review`, {
      method: 'POST',
      body: JSON.stringify(action),
    });
  }

  // Analytics endpoints
  async getNurseAnalytics(params?: {
    start_date?: string;
    end_date?: string;
  }): Promise<ApiResponse<any>> {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          searchParams.append(key, value.toString());
        }
      });
    }
    const query = searchParams.toString();
    return this.request(`/nurse/analytics${query ? `?${query}` : ''}`);
  }

  // Search PARs
  async searchPARs(query: string, filters?: {
    urgency?: string[];
    risk_level?: string[];
    has_red_flags?: boolean;
    date_from?: string;
    date_to?: string;
  }): Promise<ApiResponse<PatientAssessmentReport[]>> {
    const searchParams = new URLSearchParams({ q: query });
    if (filters) {
      Object.entries(filters).forEach(([key, value]) => {
        if (value !== undefined) {
          if (Array.isArray(value)) {
            value.forEach(v => searchParams.append(key, v));
          } else {
            searchParams.append(key, value.toString());
          }
        }
      });
    }
    return this.request(`/pars/search?${searchParams.toString()}`);
  }

  // Consent endpoints
  async getActiveConsent(): Promise<ApiResponse<ConsentInfo>> {
    return this.request('/consent/active');
  }

  async submitConsent(consentData: ConsentResponse): Promise<ApiResponse<ConsentResponse>> {
    return this.request('/consent/submit', {
      method: 'POST',
      body: JSON.stringify(consentData),
    });
  }

  // Health check
  async healthCheck(): Promise<ApiResponse<{ status: string; timestamp: string }>> {
    return this.request('/health');
  }

  // Audit logs (for nurse/admin)
  async getAuditLogs(filters?: {
    user_id?: string;
    entity_type?: string;
    start_date?: string;
    end_date?: string;
    limit?: number;
  }): Promise<ApiResponse<any[]>> {
    const params = new URLSearchParams();
    if (filters) {
      Object.entries(filters).forEach(([key, value]) => {
        if (value !== undefined) {
          params.append(key, value.toString());
        }
      });
    }
    
    const query = params.toString();
    const endpoint = `/audit/logs${query ? `?${query}` : ''}`;
    
    return this.request(endpoint);
  }
}

export const apiService = new ApiService();
export default apiService;