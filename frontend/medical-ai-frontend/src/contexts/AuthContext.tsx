import React, { createContext, useContext, useReducer, useEffect, ReactNode } from 'react';
import { User, ApiResponse } from '../types';
import { apiService } from '../services/api';

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}

type AuthAction =
  | { type: 'AUTH_START' }
  | { type: 'AUTH_SUCCESS'; payload: User }
  | { type: 'AUTH_ERROR'; payload: string }
  | { type: 'LOGOUT' }
  | { type: 'CLEAR_ERROR' };

interface AuthContextType extends AuthState {
  login: (email: string, password: string) => Promise<boolean>;
  register: (userData: {
    email: string;
    password: string;
    firstName?: string;
    lastName?: string;
  }) => Promise<boolean>;
  logout: () => void;
  clearError: () => void;
  refreshUser: () => Promise<void>;
}

const initialState: AuthState = {
  user: null,
  isAuthenticated: false,
  isLoading: true,
  error: null,
};

const authReducer = (state: AuthState, action: AuthAction): AuthState => {
  switch (action.type) {
    case 'AUTH_START':
      return {
        ...state,
        isLoading: true,
        error: null,
      };
    case 'AUTH_SUCCESS':
      return {
        ...state,
        user: action.payload,
        isAuthenticated: true,
        isLoading: false,
        error: null,
      };
    case 'AUTH_ERROR':
      return {
        ...state,
        user: null,
        isAuthenticated: false,
        isLoading: false,
        error: action.payload,
      };
    case 'LOGOUT':
      return {
        ...state,
        user: null,
        isAuthenticated: false,
        isLoading: false,
        error: null,
      };
    case 'CLEAR_ERROR':
      return {
        ...state,
        error: null,
      };
    default:
      return state;
  }
};

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [state, dispatch] = useReducer(authReducer, initialState);

  // Check for existing token on mount
  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem('authToken');
      if (token) {
        try {
          apiService.setToken(token);
          const response = await apiService.getCurrentUser();
          if (response.success && response.data) {
            dispatch({ type: 'AUTH_SUCCESS', payload: response.data });
          } else {
            // Token is invalid, clear it
            localStorage.removeItem('authToken');
            apiService.clearToken();
            dispatch({ type: 'AUTH_ERROR', payload: 'Invalid token' });
          }
        } catch (error) {
          console.error('Auth check failed:', error);
          localStorage.removeItem('authToken');
          apiService.clearToken();
          dispatch({ type: 'AUTH_ERROR', payload: 'Authentication failed' });
        }
      } else {
        dispatch({ type: 'LOGOUT' });
      }
    };

    checkAuth();
  }, []);

  const login = async (email: string, password: string): Promise<boolean> => {
    dispatch({ type: 'AUTH_START' });

    try {
      const response = await apiService.login({ email, password });
      
      if (response.success && response.data) {
        const { user, access_token } = response.data;
        apiService.setToken(access_token);
        dispatch({ type: 'AUTH_SUCCESS', payload: user });
        return true;
      } else {
        const errorMessage = response.error?.message || 'Login failed';
        dispatch({ type: 'AUTH_ERROR', payload: errorMessage });
        return false;
      }
    } catch (error: any) {
      const errorMessage = error.message || 'Login failed';
      dispatch({ type: 'AUTH_ERROR', payload: errorMessage });
      return false;
    }
  };

  const register = async (userData: {
    email: string;
    password: string;
    firstName?: string;
    lastName?: string;
  }): Promise<boolean> => {
    dispatch({ type: 'AUTH_START' });

    try {
      const response = await apiService.register({
        email: userData.email,
        password: userData.password,
        confirmPassword: userData.password,
        firstName: userData.firstName,
        lastName: userData.lastName,
      });

      if (response.success && response.data) {
        const { user, access_token } = response.data;
        apiService.setToken(access_token);
        dispatch({ type: 'AUTH_SUCCESS', payload: user });
        return true;
      } else {
        const errorMessage = response.error?.message || 'Registration failed';
        dispatch({ type: 'AUTH_ERROR', payload: errorMessage });
        return false;
      }
    } catch (error: any) {
      const errorMessage = error.message || 'Registration failed';
      dispatch({ type: 'AUTH_ERROR', payload: errorMessage });
      return false;
    }
  };

  const logout = () => {
    apiService.clearToken();
    dispatch({ type: 'LOGOUT' });
  };

  const clearError = () => {
    dispatch({ type: 'CLEAR_ERROR' });
  };

  const refreshUser = async (): Promise<void> => {
    try {
      const response = await apiService.getCurrentUser();
      if (response.success && response.data) {
        dispatch({ type: 'AUTH_SUCCESS', payload: response.data });
      }
    } catch (error) {
      console.error('Failed to refresh user:', error);
      logout();
    }
  };

  const value: AuthContextType = {
    ...state,
    login,
    register,
    logout,
    clearError,
    refreshUser,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};