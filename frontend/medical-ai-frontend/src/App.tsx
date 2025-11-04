import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import { ChatProvider } from './contexts/ChatContext';
import ChatContainer from './components/chat/ChatContainer';
import AuthPage from './pages/AuthPage';
import NurseDashboard from './pages/NurseDashboard';
import AdminDashboard from './pages/AdminDashboard';
import { useAuth } from './contexts/AuthContext';
import { Toaster } from '@/components/ui/toaster';
import LoadingScreen from './components/ui/LoadingScreen';
import './App.css';

// Protected Route Component
interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredRole?: 'patient' | 'nurse' | 'admin';
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ 
  children, 
  requiredRole 
}) => {
  const { isAuthenticated, user, isLoading } = useAuth();

  if (isLoading) {
    return <LoadingScreen message="Checking authentication..." />;
  }

  if (!isAuthenticated) {
    return <Navigate to="/auth" replace />;
  }

  if (requiredRole && user?.role !== requiredRole) {
    // Redirect to appropriate dashboard based on role
    if (user?.role === 'nurse') {
      return <Navigate to="/nurse/queue" replace />;
    } else if (user?.role === 'admin') {
      return <Navigate to="/admin" replace />;
    } else {
      return <Navigate to="/" replace />;
    }
  }

  return <>{children}</>;
};

// App Routes Component
const AppRoutes: React.FC = () => {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return <LoadingScreen message="Initializing application..." />;
  }

  return (
    <Routes>
      {/* Public Routes */}
      <Route 
        path="/auth" 
        element={
          isAuthenticated ? <Navigate to="/" replace /> : <AuthPage />
        } 
      />

      {/* Protected Patient Routes */}
      <Route 
        path="/" 
        element={
          <ProtectedRoute requiredRole="patient">
            <ChatProvider>
              <ChatContainer />
            </ChatProvider>
          </ProtectedRoute>
        } 
      />

      {/* Nurse Dashboard Routes */}
      <Route 
        path="/nurse/*" 
        element={
          <ProtectedRoute requiredRole="nurse">
            <NurseDashboard />
          </ProtectedRoute>
        } 
      />

      {/* Admin Dashboard Routes */}
      <Route 
        path="/admin" 
        element={
          <ProtectedRoute requiredRole="admin">
            <AdminDashboard />
          </ProtectedRoute>
        } 
      />

      {/* Catch all - redirect to home */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
};

// Main App Component
const App: React.FC = () => {
  return (
    <AuthProvider>
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
        <AppRoutes />
        <Toaster />
      </div>
    </AuthProvider>
  );
};

export default App;
