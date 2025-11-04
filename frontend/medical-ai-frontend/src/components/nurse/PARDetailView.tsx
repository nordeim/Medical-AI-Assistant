import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  AlertTriangle, 
  CheckCircle, 
  XCircle,
  Clock,
  FileText,
  User,
  Activity,
  BookOpen,
  ArrowLeft,
  Save
} from 'lucide-react';
import { PatientAssessmentReport, NurseAction } from '@/types';
import { apiService } from '@/services/api';
import { formatDistanceToNow, format } from 'date-fns';
import { useToast } from '@/hooks/use-toast';

const PARDetailView: React.FC = () => {
  const { parId } = useParams<{ parId: string }>();
  const navigate = useNavigate();
  const { toast } = useToast();
  
  const [par, setPar] = useState<PatientAssessmentReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [action, setAction] = useState<'approved' | 'override' | null>(null);
  const [notes, setNotes] = useState('');
  const [overrideReason, setOverrideReason] = useState('');

  useEffect(() => {
    if (parId) {
      loadPARDetail();
    }
  }, [parId]);

  const loadPARDetail = async () => {
    try {
      setLoading(true);
      const response = await apiService.getPAR(parId!);
      if (response.success && response.data) {
        setPar(response.data);
      }
    } catch (err) {
      console.error('Failed to load PAR:', err);
      toast({
        title: 'Error',
        description: 'Failed to load patient assessment report',
        variant: 'destructive'
      });
    } finally {
      setLoading(false);
    }
  };

  const handleSubmitAction = async () => {
    if (!action || !parId) return;

    try {
      setSubmitting(true);
      const actionData: NurseAction = {
        action,
        notes: notes || undefined,
        override_reason: action === 'override' ? overrideReason : undefined
      };

      const response = await apiService.submitNurseAction(parId, actionData);
      
      if (response.success) {
        toast({
          title: 'Success',
          description: `PAR ${action === 'approved' ? 'approved' : 'overridden'} successfully`,
        });
        navigate('/nurse/queue');
      }
    } catch (err) {
      console.error('Failed to submit action:', err);
      toast({
        title: 'Error',
        description: 'Failed to submit review',
        variant: 'destructive'
      });
    } finally {
      setSubmitting(false);
    }
  };

  const getRiskLevelColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'critical':
        return 'bg-red-600 text-white border-red-700';
      case 'high':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'low':
        return 'bg-green-100 text-green-800 border-green-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <Activity className="w-12 h-12 animate-spin mx-auto mb-4 text-blue-600" />
          <p className="text-gray-600">Loading assessment report...</p>
        </div>
      </div>
    );
  }

  if (!par) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <XCircle className="w-12 h-12 mx-auto mb-4 text-red-600" />
          <p className="text-gray-600">Assessment report not found</p>
          <Button onClick={() => navigate('/nurse/queue')} className="mt-4">
            Back to Queue
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Button
            variant="outline"
            onClick={() => navigate('/nurse/queue')}
            size="sm"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Queue
          </Button>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Patient Assessment Review</h1>
            <p className="text-sm text-gray-600">
              Created {formatDistanceToNow(new Date(par.created_at), { addSuffix: true })}
            </p>
          </div>
        </div>
      </div>

      {/* Status Badges */}
      <div className="flex items-center space-x-3">
        <Badge className={getRiskLevelColor(par.risk_level)}>
          {par.risk_level.toUpperCase()} RISK
        </Badge>
        <Badge className="bg-blue-100 text-blue-800 border-blue-200">
          {par.urgency.toUpperCase()}
        </Badge>
        {par.has_red_flags && (
          <Badge className="bg-red-100 text-red-800 border-red-200">
            <AlertTriangle className="w-3 h-3 mr-1" />
            RED FLAGS
          </Badge>
        )}
        {par.review_status && (
          <Badge className="bg-gray-100 text-gray-800 border-gray-200">
            {par.review_status.toUpperCase()}
          </Badge>
        )}
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - PAR Details */}
        <div className="lg:col-span-2 space-y-6">
          <Tabs defaultValue="assessment" className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="assessment">Assessment</TabsTrigger>
              <TabsTrigger value="chat">Chat Transcript</TabsTrigger>
              <TabsTrigger value="sources">RAG Sources</TabsTrigger>
              <TabsTrigger value="guidelines">Guidelines</TabsTrigger>
            </TabsList>

            <TabsContent value="assessment" className="space-y-4">
              <Card className="p-6">
                <h3 className="font-semibold text-lg mb-4 flex items-center">
                  <FileText className="w-5 h-5 mr-2" />
                  Chief Complaint
                </h3>
                <p className="text-gray-700">{par.chief_complaint}</p>
              </Card>

              <Card className="p-6">
                <h3 className="font-semibold text-lg mb-4">Symptoms</h3>
                <ul className="space-y-2">
                  {par.symptoms.map((symptom, idx) => (
                    <li key={idx} className="flex items-start">
                      <span className="w-2 h-2 bg-blue-600 rounded-full mt-2 mr-3" />
                      <span className="text-gray-700">{symptom}</span>
                    </li>
                  ))}
                </ul>
              </Card>

              {par.red_flags && par.red_flags.length > 0 && (
                <Card className="p-6 border-red-200 bg-red-50">
                  <h3 className="font-semibold text-lg mb-4 text-red-900 flex items-center">
                    <AlertTriangle className="w-5 h-5 mr-2" />
                    Red Flags
                  </h3>
                  <ul className="space-y-2">
                    {par.red_flags.map((flag, idx) => (
                      <li key={idx} className="flex items-start">
                        <AlertTriangle className="w-4 h-4 text-red-600 mt-1 mr-3" />
                        <span className="text-red-900">{flag}</span>
                      </li>
                    ))}
                  </ul>
                </Card>
              )}

              <Card className="p-6">
                <h3 className="font-semibold text-lg mb-4">AI Recommendations</h3>
                <ul className="space-y-2">
                  {par.recommendations.map((rec, idx) => (
                    <li key={idx} className="flex items-start">
                      <CheckCircle className="w-4 h-4 text-green-600 mt-1 mr-3 flex-shrink-0" />
                      <span className="text-gray-700">{rec}</span>
                    </li>
                  ))}
                </ul>
              </Card>

              {par.differential_diagnosis && par.differential_diagnosis.length > 0 && (
                <Card className="p-6">
                  <h3 className="font-semibold text-lg mb-4">Differential Diagnosis</h3>
                  <ul className="space-y-2">
                    {par.differential_diagnosis.map((diagnosis, idx) => (
                      <li key={idx} className="flex items-start">
                        <span className="text-blue-600 font-medium mr-3">{idx + 1}.</span>
                        <span className="text-gray-700">{diagnosis}</span>
                      </li>
                    ))}
                  </ul>
                </Card>
              )}
            </TabsContent>

            <TabsContent value="chat">
              <Card className="p-6">
                <h3 className="font-semibold text-lg mb-4">Chat Transcript</h3>
                <p className="text-gray-600">Chat transcript will be loaded here</p>
                <p className="text-sm text-gray-500 mt-2">Session ID: {par.session_id}</p>
              </Card>
            </TabsContent>

            <TabsContent value="sources">
              <div className="space-y-4">
                {par.rag_sources && par.rag_sources.length > 0 ? (
                  par.rag_sources.map((source, idx) => (
                    <Card key={idx} className="p-6">
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex items-center space-x-2">
                          <BookOpen className="w-5 h-5 text-blue-600" />
                          <h4 className="font-semibold text-gray-900">{source.title}</h4>
                        </div>
                        <Badge variant="outline">
                          {(source.relevance_score * 100).toFixed(0)}% match
                        </Badge>
                      </div>
                      <p className="text-gray-700 text-sm mb-2">{source.content}</p>
                      <div className="flex items-center text-xs text-gray-500">
                        <span className="capitalize">{source.source_type}</span>
                        {source.reference_url && (
                          <>
                            <span className="mx-2">•</span>
                            <a 
                              href={source.reference_url} 
                              target="_blank" 
                              rel="noopener noreferrer"
                              className="text-blue-600 hover:underline"
                            >
                              View Source
                            </a>
                          </>
                        )}
                      </div>
                    </Card>
                  ))
                ) : (
                  <Card className="p-6 text-center text-gray-600">
                    No RAG sources available for this assessment
                  </Card>
                )}
              </div>
            </TabsContent>

            <TabsContent value="guidelines">
              <div className="space-y-4">
                {par.guideline_references && par.guideline_references.length > 0 ? (
                  par.guideline_references.map((guideline, idx) => (
                    <Card key={idx} className="p-6">
                      <div className="flex items-start space-x-3 mb-3">
                        <BookOpen className="w-5 h-5 text-green-600 mt-1" />
                        <div className="flex-1">
                          <h4 className="font-semibold text-gray-900 mb-1">{guideline.title}</h4>
                          <p className="text-sm text-gray-600 mb-2">Section: {guideline.section}</p>
                          <p className="text-gray-700 mb-2">{guideline.recommendation}</p>
                          {guideline.evidence_level && (
                            <Badge variant="outline" className="text-xs">
                              Evidence: {guideline.evidence_level}
                            </Badge>
                          )}
                        </div>
                      </div>
                    </Card>
                  ))
                ) : (
                  <Card className="p-6 text-center text-gray-600">
                    No guideline references available for this assessment
                  </Card>
                )}
              </div>
            </TabsContent>
          </Tabs>
        </div>

        {/* Right Column - Action Panel */}
        <div className="space-y-6">
          <Card className="p-6">
            <h3 className="font-semibold text-lg mb-4 flex items-center">
              <User className="w-5 h-5 mr-2" />
              Review Action
            </h3>

            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <Button
                  variant={action === 'approved' ? 'default' : 'outline'}
                  onClick={() => setAction('approved')}
                  className="w-full"
                >
                  <CheckCircle className="w-4 h-4 mr-2" />
                  Approve
                </Button>
                <Button
                  variant={action === 'override' ? 'default' : 'outline'}
                  onClick={() => setAction('override')}
                  className="w-full"
                >
                  <XCircle className="w-4 h-4 mr-2" />
                  Override
                </Button>
              </div>

              {action === 'override' && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Override Reason *
                  </label>
                  <Textarea
                    value={overrideReason}
                    onChange={(e) => setOverrideReason(e.target.value)}
                    placeholder="Please provide a detailed reason for overriding the AI assessment..."
                    rows={4}
                    required
                  />
                </div>
              )}

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Clinical Notes
                </label>
                <Textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="Add any additional clinical notes or observations..."
                  rows={4}
                />
              </div>

              <Button
                onClick={handleSubmitAction}
                disabled={!action || (action === 'override' && !overrideReason) || submitting}
                className="w-full"
              >
                {submitting ? (
                  <>
                    <Activity className="w-4 h-4 mr-2 animate-spin" />
                    Submitting...
                  </>
                ) : (
                  <>
                    <Save className="w-4 h-4 mr-2" />
                    Submit Review
                  </>
                )}
              </Button>
            </div>
          </Card>

          {/* Metadata Card */}
          <Card className="p-6">
            <h3 className="font-semibold text-lg mb-4">Assessment Info</h3>
            <div className="space-y-3 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-gray-600">Created</span>
                <span className="text-gray-900">
                  {format(new Date(par.created_at), 'PPp')}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-600">Session ID</span>
                <span className="text-gray-900 font-mono text-xs">
                  {par.session_id.slice(0, 8)}...
                </span>
              </div>
              {par.confidence_scores && (
                <div className="pt-3 border-t">
                  <p className="text-gray-600 mb-2">Confidence Scores</p>
                  {Object.entries(par.confidence_scores).map(([key, value]) => (
                    <div key={key} className="flex items-center justify-between mb-1">
                      <span className="text-gray-700 text-xs capitalize">{key}</span>
                      <span className="text-gray-900 text-xs">{(value * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default PARDetailView;
