/**
 * User Feedback Collection and Analysis System
 * Clinical effectiveness and usability metrics tracking
 */

import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { Input } from '../ui/input';
import { Textarea } from '../ui/textarea';
import { Alert, AlertDescription } from '../ui/alert';

// Feedback types
interface FeedbackSubmission {
  id: string;
  userId: string;
  userRole: 'nurse' | 'physician' | 'admin' | 'technician';
  category: 'usability' | 'clinical-effectiveness' | 'technical' | 'safety' | 'workflow';
  priority: 'low' | 'medium' | 'high' | 'critical';
  rating: {
    overall: number; // 1-5
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

// Clinical effectiveness metrics
interface ClinicalMetrics {
  responseTime: number; // seconds
  accuracyScore: number; // percentage
  workflowEfficiency: number; // time saved in minutes
  errorReduction: number; // percentage
  userSatisfaction: number; // 1-5 scale
  adoptionRate: number; // percentage
  taskCompletionRate: number; // percentage
}

// Usability metrics
interface UsabilityMetrics {
  taskSuccessRate: number; // percentage
  timeOnTask: number; // seconds
  errorRate: number; // percentage
  userEngagement: number; // time spent, interactions
  learnabilityScore: number; // 1-5 scale
  memoryLoad: number; // cognitive load assessment
  satisfactionScore: number; // 1-5 scale
}

// Analysis data structure
interface FeedbackAnalysis {
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

// Feedback Modal Component
const FeedbackModal: React.FC<{
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (feedback: Omit<FeedbackSubmission, 'id' | 'timestamp' | 'status'>) => void;
  context: Partial<FeedbackSubmission['context']>;
}> = ({ isOpen, onClose, onSubmit, context }) => {
  const [feedback, setFeedback] = useState({
    userRole: 'nurse' as FeedbackSubmission['userRole'],
    category: 'usability' as FeedbackSubmission['category'],
    priority: 'medium' as FeedbackSubmission['priority'],
    rating: {
      overall: 3,
      usability: 3,
      clinicalValue: 3,
      technicalPerformance: 3,
      workflowIntegration: 3
    },
    comments: {
      positive: '',
      negative: '',
      suggestions: ''
    }
  });

  const [currentTab, setCurrentTab] = useState('rating');

  if (!isOpen) return null;

  const handleSubmit = () => {
    onSubmit({
      userId: 'current-user', // In real app, get from auth context
      ...feedback,
      context: {
        deviceType: context.deviceType || 'desktop',
        browserType: context.browserType || navigator.userAgent,
        sessionDuration: context.sessionDuration || 0,
        tasksCompleted: context.tasksCompleted || 0,
        errorCount: context.errorCount || 0
      },
      followUpRequired: feedback.priority === 'critical'
    });
    onClose();
  };

  const RatingSlider: React.FC<{
    label: string;
    value: number;
    onChange: (value: number) => void;
  }> = ({ label, value, onChange }) => (
    <div className="space-y-2">
      <div className="flex justify-between">
        <label className="text-sm font-medium">{label}</label>
        <span className="text-sm text-gray-600">{value}/5</span>
      </div>
      <input
        type="range"
        min="1"
        max="5"
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value))}
        className="w-full"
      />
      <div className="flex justify-between text-xs text-gray-500">
        <span>Poor</span>
        <span>Excellent</span>
      </div>
    </div>
  );

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <Card className="w-full max-w-2xl max-h-[90vh] overflow-auto">
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Provide Feedback</span>
            <Button variant="outline" size="sm" onClick={onClose}>
              ✕
            </Button>
          </CardTitle>
        </CardHeader>
        
        <CardContent className="space-y-6">
          <Tabs value={currentTab} onValueChange={setCurrentTab}>
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="rating">Ratings</TabsTrigger>
              <TabsTrigger value="comments">Comments</TabsTrigger>
              <TabsTrigger value="context">Context</TabsTrigger>
            </TabsList>

            {/* Ratings Tab */}
            <TabsContent value="rating" className="space-y-6">
              <div className="space-y-4">
                <RatingSlider
                  label="Overall Experience"
                  value={feedback.rating.overall}
                  onChange={(value) => setFeedback(prev => ({
                    ...prev,
                    rating: { ...prev.rating, overall: value }
                  }))}
                />
                
                <RatingSlider
                  label="Usability"
                  value={feedback.rating.usability}
                  onChange={(value) => setFeedback(prev => ({
                    ...prev,
                    rating: { ...prev.rating, usability: value }
                  }))}
                />
                
                <RatingSlider
                  label="Clinical Value"
                  value={feedback.rating.clinicalValue}
                  onChange={(value) => setFeedback(prev => ({
                    ...prev,
                    rating: { ...prev.rating, clinicalValue: value }
                  }))}
                />
                
                <RatingSlider
                  label="Technical Performance"
                  value={feedback.rating.technicalPerformance}
                  onChange={(value) => setFeedback(prev => ({
                    ...prev,
                    rating: { ...prev.rating, technicalPerformance: value }
                  }))}
                />
                
                <RatingSlider
                  label="Workflow Integration"
                  value={feedback.rating.workflowIntegration}
                  onChange={(value) => setFeedback(prev => ({
                    ...prev,
                    rating: { ...prev.rating, workflowIntegration: value }
                  }))}
                />
              </div>
            </TabsContent>

            {/* Comments Tab */}
            <TabsContent value="comments" className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">What worked well?</label>
                <Textarea
                  placeholder="Describe positive aspects of your experience..."
                  value={feedback.comments.positive}
                  onChange={(e) => setFeedback(prev => ({
                    ...prev,
                    comments: { ...prev.comments, positive: e.target.value }
                  }))}
                  rows={3}
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">What needs improvement?</label>
                <Textarea
                  placeholder="Describe issues or areas for improvement..."
                  value={feedback.comments.negative}
                  onChange={(e) => setFeedback(prev => ({
                    ...prev,
                    comments: { ...prev.comments, negative: e.target.value }
                  }))}
                  rows={3}
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">Suggestions</label>
                <Textarea
                  placeholder="How can we make this better?"
                  value={feedback.comments.suggestions}
                  onChange={(e) => setFeedback(prev => ({
                    ...prev,
                    comments: { ...prev.comments, suggestions: e.target.value }
                  }))}
                  rows={3}
                />
              </div>
            </TabsContent>

            {/* Context Tab */}
            <TabsContent value="context" className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Your Role</label>
                  <select
                    value={feedback.userRole}
                    onChange={(e) => setFeedback(prev => ({
                      ...prev,
                      userRole: e.target.value as FeedbackSubmission['userRole']
                    }))}
                    className="w-full border rounded px-3 py-2"
                  >
                    <option value="nurse">Nurse</option>
                    <option value="physician">Physician</option>
                    <option value="admin">Administrator</option>
                    <option value="technician">Technician</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">Feedback Category</label>
                  <select
                    value={feedback.category}
                    onChange={(e) => setFeedback(prev => ({
                      ...prev,
                      category: e.target.value as FeedbackSubmission['category']
                    }))}
                    className="w-full border rounded px-3 py-2"
                  >
                    <option value="usability">Usability</option>
                    <option value="clinical-effectiveness">Clinical Effectiveness</option>
                    <option value="technical">Technical</option>
                    <option value="safety">Safety</option>
                    <option value="workflow">Workflow</option>
                  </select>
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">Priority Level</label>
                <div className="grid grid-cols-4 gap-2">
                  {[
                    { value: 'low', label: 'Low', color: 'bg-green-100 text-green-800' },
                    { value: 'medium', label: 'Medium', color: 'bg-yellow-100 text-yellow-800' },
                    { value: 'high', label: 'High', color: 'bg-orange-100 text-orange-800' },
                    { value: 'critical', label: 'Critical', color: 'bg-red-100 text-red-800' }
                  ].map(option => (
                    <Button
                      key={option.value}
                      variant={feedback.priority === option.value ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => setFeedback(prev => ({
                        ...prev,
                        priority: option.value as FeedbackSubmission['priority']
                      }))}
                      className={feedback.priority === option.value ? option.color : ''}
                    >
                      {option.label}
                    </Button>
                  ))}
                </div>
              </div>
            </TabsContent>
          </Tabs>
          
          <div className="flex justify-end space-x-2 pt-4">
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button onClick={handleSubmit}>
              Submit Feedback
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

// Analytics Dashboard Component
const FeedbackAnalytics: React.FC<{
  analysis: FeedbackAnalysis;
  onExport: () => void;
}> = ({ analysis, onExport }) => {
  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-blue-600">{analysis.totalSubmissions}</div>
            <div className="text-sm text-gray-600">Total Feedback</div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-green-600">{analysis.averageRating.toFixed(1)}</div>
            <div className="text-sm text-gray-600">Average Rating</div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-orange-600">{analysis.priorityDistribution.high + analysis.priorityDistribution.critical}</div>
            <div className="text-sm text-gray-600">High Priority Issues</div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-purple-600">{analysis.recommendations.length}</div>
            <div className="text-sm text-gray-600">Recommendations</div>
          </CardContent>
        </Card>
      </div>

      {/* Category Distribution */}
      <Card>
        <CardHeader>
          <CardTitle>Feedback by Category</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {Object.entries(analysis.categoryDistribution).map(([category, count]) => (
              <div key={category} className="flex items-center justify-between">
                <span className="capitalize">{category.replace('-', ' ')}</span>
                <div className="flex items-center space-x-2">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full"
                      style={{ width: `${(count / analysis.totalSubmissions) * 100}%` }}
                    />
                  </div>
                  <Badge variant="outline">{count}</Badge>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Top Issues */}
      <Card>
        <CardHeader>
          <CardTitle>Top Issues</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {analysis.topIssues.map((issue, index) => (
              <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                <div className="flex-1">
                  <div className="font-medium">{issue.issue}</div>
                  <div className="text-sm text-gray-600">Reported {issue.frequency} times</div>
                </div>
                <div className="flex items-center space-x-2">
                  <Badge variant={issue.impact === 'high' ? 'destructive' : issue.impact === 'medium' ? 'secondary' : 'outline'}>
                    {issue.impact} impact
                  </Badge>
                  <Badge variant={issue.status === 'resolved' ? 'default' : issue.status === 'in-progress' ? 'secondary' : 'outline'}>
                    {issue.status}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Clinical Improvements */}
      <Card>
        <CardHeader>
          <CardTitle>Recommended Clinical Improvements</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {analysis.clinicalImprovements.map((improvement, index) => (
              <div key={index} className="p-4 border rounded-lg">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h4 className="font-semibold">{improvement.area}</h4>
                    <p className="text-sm text-gray-600 mt-1">{improvement.improvement}</p>
                    <p className="text-sm text-blue-600 mt-1">{improvement.expectedImpact}</p>
                  </div>
                  <Badge variant="outline">{improvement.effort} effort</Badge>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

// Main Feedback Collection Component
export const UserFeedbackCollection: React.FC<{
  userId: string;
  userRole: 'nurse' | 'physician' | 'admin' | 'technician';
  onFeedbackSubmit: (feedback: FeedbackSubmission) => void;
}> = ({ userId, userRole, onFeedbackSubmit }) => {
  const [feedbackSubmissions, setFeedbackSubmissions] = useState<FeedbackSubmission[]>([]);
  const [isFeedbackModalOpen, setIsFeedbackModalOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('submit');
  const [context, setContext] = useState<Partial<FeedbackSubmission['context']>>({
    deviceType: 'desktop',
    browserType: navigator.userAgent,
    sessionDuration: 0,
    tasksCompleted: 0,
    errorCount: 0
  });

  const sessionStartTime = useRef<Date>(new Date());

  // Track session metrics
  useEffect(() => {
    const interval = setInterval(() => {
      const duration = Math.floor((new Date().getTime() - sessionStartTime.current.getTime()) / 1000);
      setContext(prev => ({ ...prev, sessionDuration: duration }));
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // Generate mock analysis data
  const analysis: FeedbackAnalysis = {
    totalSubmissions: feedbackSubmissions.length,
    averageRating: feedbackSubmissions.length > 0 
      ? feedbackSubmissions.reduce((sum, sub) => sum + sub.rating.overall, 0) / feedbackSubmissions.length
      : 0,
    categoryDistribution: {
      'usability': feedbackSubmissions.filter(s => s.category === 'usability').length,
      'clinical-effectiveness': feedbackSubmissions.filter(s => s.category === 'clinical-effectiveness').length,
      'technical': feedbackSubmissions.filter(s => s.category === 'technical').length,
      'safety': feedbackSubmissions.filter(s => s.category === 'safety').length,
      'workflow': feedbackSubmissions.filter(s => s.category === 'workflow').length
    },
    priorityDistribution: {
      'low': feedbackSubmissions.filter(s => s.priority === 'low').length,
      'medium': feedbackSubmissions.filter(s => s.priority === 'medium').length,
      'high': feedbackSubmissions.filter(s => s.priority === 'high').length,
      'critical': feedbackSubmissions.filter(s => s.priority === 'critical').length
    },
    trendAnalysis: {
      weeklyVolume: [12, 19, 15, 23, 28, 25, 31],
      ratingTrends: [3.2, 3.4, 3.6, 3.8, 3.9, 4.1, 4.2],
      categoryTrends: {
        'usability': [3, 5, 4, 6, 8, 7, 9],
        'clinical-effectiveness': [4, 3, 5, 4, 6, 8, 7],
        'technical': [2, 4, 3, 5, 4, 6, 5]
      }
    },
    topIssues: [
      {
        issue: 'Patient chat response delays during peak hours',
        frequency: 15,
        impact: 'high',
        status: 'in-progress'
      },
      {
        issue: 'Mobile interface buttons too small for touch interaction',
        frequency: 12,
        impact: 'medium',
        status: 'open'
      },
      {
        issue: 'Emergency alert notifications not reaching all devices',
        frequency: 8,
        impact: 'high',
        status: 'resolved'
      }
    ],
    recommendations: [
      'Implement edge caching for improved response times',
      'Enhance mobile touch targets and add haptic feedback',
      'Add redundancy to emergency notification system',
      'Improve accessibility for screen reader users',
      'Streamline clinical documentation workflow'
    ],
    clinicalImprovements: [
      {
        area: 'Patient Communication',
        improvement: 'Add structured templates for common medical scenarios',
        expectedImpact: 'Reduce documentation time by 30%',
        effort: 'medium'
      },
      {
        area: 'Emergency Response',
        improvement: 'Implement predictive alerts for patient deterioration',
        expectedImpact: 'Reduce response time by 45%',
        effort: 'high'
      },
      {
        area: 'Workflow Integration',
        improvement: 'Add one-click EMR synchronization',
        expectedImpact: 'Eliminate duplicate data entry',
        effort: 'low'
      }
    ]
  };

  const handleFeedbackSubmit = (newFeedback: Omit<FeedbackSubmission, 'id' | 'timestamp' | 'status'>) => {
    const feedback: FeedbackSubmission = {
      ...newFeedback,
      id: Date.now().toString(),
      timestamp: new Date(),
      status: 'new'
    };

    setFeedbackSubmissions(prev => [feedback, ...prev]);
    onFeedbackSubmit(feedback);
  };

  const handleExport = () => {
    const data = JSON.stringify(feedbackSubmissions, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `feedback-analysis-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="h-full bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Feedback & Analytics</h1>
            <p className="text-gray-600">Help us improve your medical workflow experience</p>
          </div>
          <div className="flex space-x-2">
            <Button
              onClick={() => setIsFeedbackModalOpen(true)}
              className="bg-blue-600 hover:bg-blue-700"
            >
              📝 Provide Feedback
            </Button>
            {feedbackSubmissions.length > 0 && (
              <Button variant="outline" onClick={handleExport}>
                📊 Export Data
              </Button>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="p-6">
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="submit">Submit Feedback</TabsTrigger>
            <TabsTrigger value="analytics">Analytics</TabsTrigger>
            <TabsTrigger value="my-feedback">My Submissions</TabsTrigger>
          </TabsList>

          {/* Submit Feedback Tab */}
          <TabsContent value="submit" className="space-y-6">
            <Alert>
              <AlertDescription>
                <strong>Your feedback matters!</strong> Help us improve clinical workflows, 
                patient safety, and medical technology effectiveness.
              </AlertDescription>
            </Alert>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Quick Feedback</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <p className="text-gray-600">
                    Share your immediate thoughts about your current session experience.
                  </p>
                  <div className="space-y-2">
                    {[
                      { label: 'Great experience', icon: '👍' },
                      { label: 'Needs improvement', icon: '👎' },
                      { label: 'Critical issue', icon: '🚨' },
                      { label: 'Feature request', icon: '💡' }
                    ].map(option => (
                      <Button
                        key={option.label}
                        variant="outline"
                        className="w-full justify-start"
                        onClick={() => setIsFeedbackModalOpen(true)}
                      >
                        <span className="mr-2">{option.icon}</span>
                        {option.label}
                      </Button>
                    ))}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Session Context</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Session Duration:</span>
                      <Badge variant="outline">{Math.floor(context.sessionDuration! / 60)}m {context.sessionDuration! % 60}s</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Device Type:</span>
                      <Badge variant="outline">{context.deviceType}</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Tasks Completed:</span>
                      <Badge variant="outline">{context.tasksCompleted}</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Errors Encountered:</span>
                      <Badge variant={context.errorCount! > 0 ? 'destructive' : 'default'}>
                        {context.errorCount}
                      </Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Analytics Tab */}
          <TabsContent value="analytics" className="space-y-6">
            <FeedbackAnalytics analysis={analysis} onExport={handleExport} />
          </TabsContent>

          {/* My Submissions Tab */}
          <TabsContent value="my-feedback" className="space-y-6">
            {feedbackSubmissions.length === 0 ? (
              <Card>
                <CardContent className="p-8 text-center">
                  <div className="text-gray-400 text-4xl mb-4">📝</div>
                  <h3 className="text-lg font-semibold text-gray-600 mb-2">No submissions yet</h3>
                  <p className="text-gray-500 mb-4">
                    Share your feedback to help improve the medical workflow system
                  </p>
                  <Button onClick={() => setIsFeedbackModalOpen(true)}>
                    Provide First Feedback
                  </Button>
                </CardContent>
              </Card>
            ) : (
              <div className="space-y-4">
                {feedbackSubmissions.map(submission => (
                  <Card key={submission.id}>
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-lg capitalize">
                          {submission.category.replace('-', ' ')} Feedback
                        </CardTitle>
                        <div className="flex items-center space-x-2">
                          <Badge variant={submission.priority === 'critical' ? 'destructive' : submission.priority === 'high' ? 'secondary' : 'outline'}>
                            {submission.priority}
                          </Badge>
                          <Badge variant="outline">
                            {submission.rating.overall}/5 ⭐
                          </Badge>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {submission.comments.positive && (
                        <div>
                          <h4 className="font-semibold text-green-700 mb-1">What worked well:</h4>
                          <p className="text-gray-600 text-sm">{submission.comments.positive}</p>
                        </div>
                      )}
                      
                      {submission.comments.negative && (
                        <div>
                          <h4 className="font-semibold text-red-700 mb-1">Areas for improvement:</h4>
                          <p className="text-gray-600 text-sm">{submission.comments.negative}</p>
                        </div>
                      )}
                      
                      {submission.comments.suggestions && (
                        <div>
                          <h4 className="font-semibold text-blue-700 mb-1">Suggestions:</h4>
                          <p className="text-gray-600 text-sm">{submission.comments.suggestions}</p>
                        </div>
                      )}
                      
                      <div className="flex justify-between text-sm text-gray-500">
                        <span>Submitted: {submission.timestamp.toLocaleString()}</span>
                        <span>Status: {submission.status}</span>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>
        </Tabs>
      </div>

      {/* Feedback Modal */}
      <FeedbackModal
        isOpen={isFeedbackModalOpen}
        onClose={() => setIsFeedbackModalOpen(false)}
        onSubmit={handleFeedbackSubmit}
        context={context}
      />
    </div>
  );
};

export default UserFeedbackCollection;