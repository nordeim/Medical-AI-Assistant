/**
 * Nurse Dashboard Usability Enhancement
 * Optimized for clinical workflows with intuitive navigation and quick actions
 */

import React, { useState, useEffect, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { ScrollArea } from '../ui/scroll-area';
import { Input } from '../ui/input';
import { Alert, AlertDescription } from '../ui/alert';
import { useToast } from '../hooks/use-toast';

// Patient priority scoring system
interface Patient {
  id: string;
  name: string;
  age: number;
  condition: string;
  priority: 'critical' | 'high' | 'medium' | 'low';
  waitTime: number; // minutes
  lastContact: Date;
  notes: string;
  vitals?: {
    heartRate: number;
    bloodPressure: string;
    temperature: number;
    oxygenSaturation: number;
  };
}

// Quick action configurations
interface QuickAction {
  id: string;
  label: string;
  icon: string;
  action: () => void;
  category: 'emergency' | 'communication' | 'documentation' | 'assessment';
  keyboardShortcut?: string;
}

interface DashboardLayout {
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
  refreshInterval: number; // seconds
}

interface NurseDashboardOptimizationProps {
  patients: Patient[];
  onPatientSelect: (patientId: string) => void;
  onCriticalAlert: (patientId: string, alert: string) => void;
  layout?: Partial<DashboardLayout>;
}

export const NurseDashboardOptimization: React.FC<NurseDashboardOptimizationProps> = ({
  patients,
  onPatientSelect,
  onCriticalAlert,
  layout = {
    widgets: {
      activePatients: true,
      criticalAlerts: true,
      recentMessages: true,
      quickActions: true,
      vitalsMonitor: true,
      schedule: false
    },
    compactMode: false,
    autoRefresh: true,
    refreshInterval: 30
  }
}) => {
  const [dashboardLayout, setDashboardLayout] = useState<DashboardLayout>({
    ...layout,
    widgets: { ...layout.widgets },
    compactMode: layout.compactMode || false,
    autoRefresh: layout.autoRefresh || true,
    refreshInterval: layout.refreshInterval || 30
  });
  
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [lastRefresh, setLastRefresh] = useState(new Date());
  const { toast } = useToast();

  // Calculate patient priority scores
  const calculatePriorityScore = (patient: Patient): number => {
    const baseScore = {
      critical: 100,
      high: 70,
      medium: 40,
      low: 10
    }[patient.priority];

    // Time penalty - longer waits increase priority
    const timePenalty = Math.min(patient.waitTime / 10, 20);
    
    // Age factor - elderly patients get priority boost
    const ageBonus = patient.age > 75 ? 10 : patient.age > 65 ? 5 : 0;
    
    // Vitals bonus - abnormal vitals increase priority
    const vitalsBonus = patient.vitals ? (
      (patient.vitals.heartRate > 100 || patient.vitals.heartRate < 60 ? 15 : 0) +
      (patient.vitals.temperature > 38 || patient.vitals.temperature < 36 ? 10 : 0) +
      (patient.vitals.oxygenSaturation < 95 ? 15 : 0)
    ) : 0;

    return Math.min(baseScore + timePenalty + ageBonus + vitalsBonus, 150);
  };

  // Filter and sort patients
  const sortedPatients = useMemo(() => {
    const filtered = patients.filter(patient => {
      const matchesSearch = patient.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           patient.condition.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesCategory = selectedCategory === 'all' || patient.priority === selectedCategory;
      return matchesSearch && matchesCategory;
    });

    return filtered.sort((a, b) => calculatePriorityScore(b) - calculatePriorityScore(a));
  }, [patients, searchTerm, selectedCategory]);

  // Quick actions definition
  const quickActions: QuickAction[] = [
    {
      id: 'emergency-call',
      label: 'Emergency Call',
      icon: '🚨',
      category: 'emergency',
      keyboardShortcut: 'Ctrl+E',
      action: () => toast({ title: "Emergency Protocol", description: "Contacting emergency services..." })
    },
    {
      id: 'message-patient',
      label: 'Message Patient',
      icon: '💬',
      category: 'communication',
      keyboardShortcut: 'Ctrl+M',
      action: () => toast({ title: "Patient Communication", description: "Opening message interface..." })
    },
    {
      id: 'record-vitals',
      label: 'Record Vitals',
      icon: '📊',
      category: 'assessment',
      keyboardShortcut: 'Ctrl+V',
      action: () => toast({ title: "Vital Signs", description: "Opening vitals recording form..." })
    },
    {
      id: 'add-note',
      label: 'Add Note',
      icon: '📝',
      category: 'documentation',
      keyboardShortcut: 'Ctrl+N',
      action: () => toast({ title: "Documentation", description: "Opening clinical notes..." })
    },
    {
      id: 'schedule-followup',
      label: 'Schedule Follow-up',
      icon: '📅',
      category: 'assessment',
      keyboardShortcut: 'Ctrl+F',
      action: () => toast({ title: "Scheduling", description: "Opening appointment scheduler..." })
    }
  ];

  // Auto-refresh mechanism
  useEffect(() => {
    if (!dashboardLayout.autoRefresh) return;

    const interval = setInterval(() => {
      setLastRefresh(new Date());
      // Simulate data refresh
      toast({ title: "Data Updated", description: "Patient data refreshed automatically", duration: 2000 });
    }, dashboardLayout.refreshInterval * 1000);

    return () => clearInterval(interval);
  }, [dashboardLayout.autoRefresh, dashboardLayout.refreshInterval, toast]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.ctrlKey) {
        const action = quickActions.find(act => act.keyboardShortcut === `Ctrl+${event.key.toUpperCase()}`);
        if (action) {
          event.preventDefault();
          action.action();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Critical alerts
  const criticalPatients = sortedPatients.filter(p => p.priority === 'critical');
  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'bg-red-500';
      case 'high': return 'bg-orange-500';
      case 'medium': return 'bg-yellow-500';
      case 'low': return 'bg-green-500';
      default: return 'bg-gray-500';
    }
  };

  const getPriorityIcon = (priority: string) => {
    switch (priority) {
      case 'critical': return '🔴';
      case 'high': return '🟠';
      case 'medium': return '🟡';
      case 'low': return '🟢';
      default: return '⚪';
    }
  };

  const getTimeSinceLastContact = (lastContact: Date): string => {
    const minutes = Math.floor((new Date().getTime() - lastContact.getTime()) / 60000);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    return `${hours}h ${minutes % 60}m ago`;
  };

  return (
    <div className="h-full bg-gray-50">
      {/* Header with quick actions and controls */}
      <div className="bg-white border-b p-4">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Nurse Dashboard</h1>
            <p className="text-sm text-gray-600">
              Last updated: {lastRefresh.toLocaleTimeString()}
            </p>
          </div>
          
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setDashboardLayout(prev => ({ ...prev, compactMode: !prev.compactMode }))}
            >
              {dashboardLayout.compactMode ? '📏' : '📐'} Compact
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setLastRefresh(new Date())}
            >
              🔄 Refresh
            </Button>
          </div>
        </div>

        {/* Quick Actions Toolbar */}
        <div className="flex flex-wrap gap-2 mb-4">
          {quickActions.map(action => (
            <Button
              key={action.id}
              variant="outline"
              size="sm"
              onClick={action.action}
              className="flex items-center space-x-1"
              title={`${action.label} (${action.keyboardShortcut})`}
            >
              <span>{action.icon}</span>
              <span className={dashboardLayout.compactMode ? 'hidden sm:inline' : ''}>{action.label}</span>
            </Button>
          ))}
        </div>

        {/* Search and Filter Controls */}
        <div className="flex items-center space-x-4">
          <div className="flex-1">
            <Input
              placeholder="Search patients by name or condition..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full"
            />
          </div>
          
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="border rounded px-3 py-2 bg-white"
          >
            <option value="all">All Priorities</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>
        </div>
      </div>

      {/* Critical Alerts Banner */}
      {criticalPatients.length > 0 && (
        <Alert className="mx-4 mt-4 border-red-200 bg-red-50">
          <AlertDescription className="text-red-800">
            <strong>🚨 {criticalPatients.length} critical patient{criticalPatients.length > 1 ? 's' : ''} need immediate attention</strong>
          </AlertDescription>
        </Alert>
      )}

      {/* Main Content Area */}
      <div className="p-4">
        <div className={`grid gap-4 ${dashboardLayout.compactMode ? 'grid-cols-1' : 'grid-cols-1 lg:grid-cols-3'}`}>
          
          {/* Active Patients Widget */}
          {dashboardLayout.widgets.activePatients && (
            <Card className={`${dashboardLayout.compactMode ? '' : 'lg:col-span-2'}`}>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span>Active Patients ({sortedPatients.length})</span>
                  <Badge variant="secondary">{criticalPatients.length} Critical</Badge>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className={`${dashboardLayout.compactMode ? 'h-96' : 'h-[600px]'}`}>
                  <div className="space-y-2">
                    {sortedPatients.map(patient => (
                      <div
                        key={patient.id}
                        className={`p-3 border rounded-lg cursor-pointer transition-colors hover:bg-gray-50 ${
                          patient.priority === 'critical' ? 'border-red-300 bg-red-50' : ''
                        }`}
                        onClick={() => onPatientSelect(patient.id)}
                        role="button"
                        tabIndex={0}
                        onKeyDown={(e) => e.key === 'Enter' && onPatientSelect(patient.id)}
                        aria-label={`Select patient ${patient.name}, ${patient.condition}, priority ${patient.priority}`}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-2">
                            <div className={`w-3 h-3 rounded-full ${getPriorityColor(patient.priority)}`} />
                            <span className="font-medium">{patient.name}</span>
                            <span className="text-sm text-gray-600">({patient.age})</span>
                          </div>
                          <div className="flex items-center space-x-2">
                            <Badge variant={patient.priority === 'critical' ? 'destructive' : 'secondary'}>
                              {getPriorityIcon(patient.priority)} {patient.priority}
                            </Badge>
                          </div>
                        </div>
                        
                        <div className="mt-2 text-sm text-gray-600">
                          <p><strong>Condition:</strong> {patient.condition}</p>
                          <p><strong>Wait time:</strong> {patient.waitTime} minutes</p>
                          <p><strong>Last contact:</strong> {getTimeSinceLastContact(patient.lastContact)}</p>
                        </div>
                        
                        {patient.vitals && (
                          <div className="mt-2 flex space-x-4 text-xs">
                            <span>HR: {patient.vitals.heartRate}</span>
                            <span>BP: {patient.vitals.bloodPressure}</span>
                            <span>Temp: {patient.vitals.temperature}°C</span>
                            <span>O2: {patient.vitals.oxygenSaturation}%</span>
                          </div>
                        )}
                        
                        {patient.priority === 'critical' && (
                          <div className="mt-2">
                            <Button
                              size="sm"
                              variant="destructive"
                              onClick={(e) => {
                                e.stopPropagation();
                                onCriticalAlert(patient.id, 'Critical patient needs immediate attention');
                              }}
                            >
                              Alert Team
                            </Button>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          )}

          {/* Quick Stats Widget */}
          <Card>
            <CardHeader>
              <CardTitle>Quick Stats</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span>Total Active:</span>
                  <Badge>{patients.length}</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span>Critical:</span>
                  <Badge variant="destructive">{criticalPatients.length}</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span>High Priority:</span>
                  <Badge variant="secondary">
                    {patients.filter(p => p.priority === 'high').length}
                  </Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span>Average Wait:</span>
                  <Badge variant="outline">
                    {Math.round(patients.reduce((acc, p) => acc + p.waitTime, 0) / patients.length || 0)}m
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default NurseDashboardOptimization;