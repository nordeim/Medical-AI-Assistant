/**
 * User Onboarding and Training Module
 * Comprehensive medical workflow guidance and safety protocol training
 */

import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Progress } from '../ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { Alert, AlertDescription } from '../ui/alert';

// Training module types
interface TrainingModule {
  id: string;
  title: string;
  description: string;
  category: 'safety' | 'workflow' | 'emergency' | 'documentation' | 'communication';
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  duration: number; // minutes
  prerequisites?: string[];
  content: TrainingContent[];
  assessment: Assessment;
}

interface TrainingContent {
  type: 'text' | 'video' | 'interactive' | 'simulation' | 'quiz';
  title: string;
  content: string;
  mediaUrl?: string;
  interactiveElements?: any[];
}

interface Assessment {
  questions: Question[];
  passingScore: number;
  maxAttempts: number;
}

interface Question {
  id: string;
  type: 'multiple-choice' | 'true-false' | 'scenario' | 'simulation';
  question: string;
  options?: string[];
  correctAnswer: string | string[];
  explanation: string;
  category: 'safety' | 'emergency' | 'workflow' | 'documentation';
}

// User progress tracking
interface UserProgress {
  userId: string;
  completedModules: string[];
  currentModule: string | null;
  moduleScores: { [moduleId: string]: number };
  attempts: { [moduleId: string]: number };
  lastAccess: Date;
  certifications: string[];
}

// Medical workflow templates
interface MedicalWorkflow {
  id: string;
  name: string;
  category: 'admission' | 'discharge' | 'emergency' | 'medication' | 'vitals';
  steps: WorkflowStep[];
  safetyChecks: SafetyCheck[];
  emergencyProcedures: EmergencyProcedure[];
}

interface WorkflowStep {
  id: string;
  title: string;
  description: string;
  order: number;
  required: boolean;
  safetyCritical: boolean;
  estimatedTime: number; // seconds
  dependencies?: string[];
}

interface SafetyCheck {
  id: string;
  checkType: 'verification' | 'confirmation' | 'double-check';
  description: string;
  verification: boolean;
  autoCheck?: boolean;
}

interface EmergencyProcedure {
  trigger: string;
  steps: string[];
  escalation: string;
  notification: string[];
}

// Interactive training components
const WorkflowSimulation: React.FC<{
  workflow: MedicalWorkflow;
  onStepComplete: (stepId: string, passed: boolean) => void;
  onWorkflowComplete: (passed: boolean) => void;
}> = ({ workflow, onStepComplete, onWorkflowComplete }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [completedSteps, setCompletedSteps] = useState<string[]>([]);
  const [safetyChecks, setSafetyChecks] = useState<{ [key: string]: boolean }>({});

  const currentStepData = workflow.steps[currentStep];

  const handleStepComplete = () => {
    const stepPassed = true; // In real implementation, this would be based on actions
    setCompletedSteps(prev => [...prev, currentStepData.id]);
    onStepComplete(currentStepData.id, stepPassed);
    
    if (currentStep < workflow.steps.length - 1) {
      setCurrentStep(prev => prev + 1);
    } else {
      onWorkflowComplete(true);
    }
  };

  const handleSafetyCheck = (checkId: string, passed: boolean) => {
    setSafetyChecks(prev => ({ ...prev, [checkId]: passed }));
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>{workflow.name} Simulation</span>
          <Badge variant="outline">
            Step {currentStep + 1} of {workflow.steps.length}
          </Badge>
        </CardTitle>
        <Progress 
          value={(currentStep / workflow.steps.length) * 100} 
          className="w-full" 
        />
      </CardHeader>
      
      <CardContent className="space-y-6">
        {/* Current Step */}
        {currentStepData && (
          <div className="border rounded-lg p-4 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">{currentStepData.title}</h3>
              {currentStepData.safetyCritical && (
                <Badge variant="destructive">⚠️ Safety Critical</Badge>
              )}
            </div>
            
            <p className="text-gray-600">{currentStepData.description}</p>
            
            {currentStepData.estimatedTime && (
              <div className="text-sm text-gray-500">
                ⏱️ Estimated time: {currentStepData.estimatedTime}s
              </div>
            )}
            
            {/* Dependencies */}
            {currentStepData.dependencies && currentStepData.dependencies.length > 0 && (
              <Alert>
                <AlertDescription>
                  <strong>Dependencies:</strong> Complete these steps first: {' '}
                  {currentStepData.dependencies.join(', ')}
                </AlertDescription>
              </Alert>
            )}
          </div>
        )}
        
        {/* Safety Checks */}
        {workflow.safetyChecks.map(check => (
          <div key={check.id} className="border rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <input
                type="checkbox"
                id={check.id}
                checked={safetyChecks[check.id] || false}
                onChange={(e) => handleSafetyCheck(check.id, e.target.checked)}
              />
              <label htmlFor={check.id} className="font-medium">
                {check.description}
              </label>
            </div>
            {check.checkType === 'verification' && (
              <div className="text-sm text-gray-500 ml-6">
                This requires verification by another healthcare professional
              </div>
            )}
          </div>
        ))}
        
        {/* Emergency Procedures */}
        {workflow.emergencyProcedures.length > 0 && (
          <Alert className="border-red-200 bg-red-50">
            <AlertDescription className="text-red-800">
              <strong>Emergency Procedures Available</strong>
              <ul className="mt-2 space-y-1">
                {workflow.emergencyProcedures.map(procedure => (
                  <li key={procedure.trigger}>
                    • Trigger: {procedure.trigger}
                  </li>
                ))}
              </ul>
            </AlertDescription>
          </Alert>
        )}
        
        {/* Action Buttons */}
        <div className="flex justify-between">
          <Button
            variant="outline"
            onClick={() => setCurrentStep(prev => Math.max(0, prev - 1))}
            disabled={currentStep === 0}
          >
            ← Previous Step
          </Button>
          
          <Button
            onClick={handleStepComplete}
            disabled={currentStepData?.safetyCritical && !Object.values(safetyChecks).every(Boolean)}
          >
            {currentStep === workflow.steps.length - 1 ? 'Complete Workflow' : 'Next Step →'}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

// Main Onboarding Component
export const UserOnboardingTraining: React.FC<{
  userId: string;
  userRole: 'nurse' | 'physician' | 'admin' | 'technician';
  onProgressUpdate: (progress: UserProgress) => void;
}> = ({ userId, userRole, onProgressUpdate }) => {
  const [progress, setProgress] = useState<UserProgress>({
    userId,
    completedModules: [],
    currentModule: null,
    moduleScores: {},
    attempts: {},
    lastAccess: new Date(),
    certifications: []
  });
  
  const [activeTab, setActiveTab] = useState('onboarding');
  const [currentAssessment, setCurrentAssessment] = useState<{
    moduleId: string;
    questionIndex: number;
    answers: { [key: string]: any };
  } | null>(null);

  // Predefined training modules
  const trainingModules: TrainingModule[] = [
    {
      id: 'safety-fundamentals',
      title: 'Medical Safety Fundamentals',
      description: 'Essential safety protocols and procedures',
      category: 'safety',
      difficulty: 'beginner',
      duration: 30,
      content: [
        {
          type: 'text',
          title: 'Introduction to Patient Safety',
          content: 'Patient safety is the cornerstone of healthcare delivery...'
        },
        {
          type: 'interactive',
          title: 'Safety Scenario Simulation',
          content: 'Practice identifying safety risks in clinical scenarios'
        }
      ],
      assessment: {
        questions: [
          {
            id: 'q1',
            type: 'multiple-choice',
            question: 'What is the most important factor in patient safety?',
            options: ['Speed of care', 'Clear communication', 'Technology', 'Cost efficiency'],
            correctAnswer: 'Clear communication',
            explanation: 'Clear communication between healthcare providers and patients is fundamental to safety.',
            category: 'safety'
          },
          {
            id: 'q2',
            type: 'scenario',
            question: 'A patient mentions they are allergic to penicillin but their chart shows no allergy. What should you do?',
            correctAnswer: 'Verify the allergy immediately and update the chart',
            explanation: 'Always trust patient-reported allergies and verify/ update the medical record.',
            category: 'safety'
          }
        ],
        passingScore: 80,
        maxAttempts: 3
      }
    },
    {
      id: 'emergency-protocols',
      title: 'Emergency Response Protocols',
      description: 'Critical emergency procedures and escalation',
      category: 'emergency',
      difficulty: 'intermediate',
      duration: 45,
      prerequisites: ['safety-fundamentals'],
      content: [
        {
          type: 'video',
          title: 'Emergency Response Overview',
          content: 'Learn the standard emergency response protocols...'
        },
        {
          type: 'simulation',
          title: 'Code Blue Simulation',
          content: 'Practice responding to cardiac arrest scenarios'
        }
      ],
      assessment: {
        questions: [
          {
            id: 'q3',
            type: 'true-false',
            question: 'In a code blue situation, the first step is to call for help.',
            correctAnswer: 'true',
            explanation: 'Always call for help first, then begin CPR if trained.',
            category: 'emergency'
          }
        ],
        passingScore: 90,
        maxAttempts: 2
      }
    },
    {
      id: 'workflow-optimization',
      title: 'Clinical Workflow Optimization',
      description: 'Efficient patient care workflows and documentation',
      category: 'workflow',
      difficulty: 'intermediate',
      duration: 35,
      content: [
        {
          type: 'interactive',
          title: 'Patient Admission Workflow',
          content: 'Learn the step-by-step patient admission process'
        }
      ],
      assessment: {
        questions: [
          {
            id: 'q4',
            type: 'multiple-choice',
            question: 'Which information must be verified during patient admission?',
            options: ['Identity only', 'Identity, allergies, medications', 'Insurance information only', 'Contact details only'],
            correctAnswer: 'Identity, allergies, medications',
            explanation: 'All three must be verified for patient safety.',
            category: 'workflow'
          }
        ],
        passingScore: 85,
        maxAttempts: 3
      }
    }
  ];

  // Medical workflows for simulation
  const medicalWorkflows: MedicalWorkflow[] = [
    {
      id: 'patient-admission',
      name: 'Patient Admission Process',
      category: 'admission',
      steps: [
        {
          id: 'verify-identity',
          title: 'Verify Patient Identity',
          description: 'Confirm patient identity using two identifiers',
          order: 1,
          required: true,
          safetyCritical: true,
          estimatedTime: 120
        },
        {
          id: 'allergy-check',
          title: 'Check Allergies',
          description: 'Verify and document any known allergies',
          order: 2,
          required: true,
          safetyCritical: true,
          estimatedTime: 60
        },
        {
          id: 'vitals-assessment',
          title: 'Initial Vitals',
          description: 'Record initial vital signs and assessments',
          order: 3,
          required: true,
          safetyCritical: false,
          estimatedTime: 300
        }
      ],
      safetyChecks: [
        {
          id: 'identity-verification',
          checkType: 'verification',
          description: 'Patient identity confirmed with two identifiers',
          verification: true,
          autoCheck: false
        }
      ],
      emergencyProcedures: [
        {
          trigger: 'Patient deterioration detected',
          steps: ['Assess patient', 'Call for help', 'Initiate emergency protocol'],
          escalation: 'Physician immediately',
          notification: ['Emergency team', 'Primary physician']
        }
      ]
    }
  ];

  // Start assessment
  const startAssessment = (moduleId: string) => {
    setCurrentAssessment({
      moduleId,
      questionIndex: 0,
      answers: {}
    });
  };

  // Submit answer
  const submitAnswer = (questionId: string, answer: string) => {
    if (!currentAssessment) return;
    
    setCurrentAssessment(prev => ({
      ...prev!,
      answers: {
        ...prev!.answers,
        [questionId]: answer
      }
    }));
  };

  // Complete assessment
  const completeAssessment = () => {
    if (!currentAssessment) return;

    const module = trainingModules.find(m => m.id === currentAssessment.moduleId);
    if (!module) return;

    let score = 0;
    module.assessment.questions.forEach(question => {
      const userAnswer = currentAssessment.answers[question.id];
      if (Array.isArray(question.correctAnswer)) {
        if (JSON.stringify(userAnswer?.sort()) === JSON.stringify(question.correctAnswer.sort())) {
          score += 1;
        }
      } else if (userAnswer === question.correctAnswer) {
        score += 1;
      }
    });

    const percentage = (score / module.assessment.questions.length) * 100;
    const passed = percentage >= module.assessment.passingScore;

    // Update progress
    const newProgress = {
      ...progress,
      moduleScores: {
        ...progress.moduleScores,
        [currentAssessment.moduleId]: percentage
      },
      attempts: {
        ...progress.attempts,
        [currentAssessment.moduleId]: (progress.attempts[currentAssessment.moduleId] || 0) + 1
      }
    };

    if (passed) {
      newProgress.completedModules.push(currentAssessment.moduleId);
      if (!newProgress.certifications.includes(currentAssessment.moduleId)) {
        newProgress.certifications.push(currentAssessment.moduleId);
      }
    }

    setProgress(newProgress);
    onProgressUpdate(newProgress);
    setCurrentAssessment(null);
  };

  return (
    <div className="h-full bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              {userRole === 'nurse' ? 'Nurse' : userRole === 'physician' ? 'Physician' : 'Medical'} Training Center
            </h1>
            <p className="text-gray-600">Welcome to your medical workflow training</p>
          </div>
          <div className="flex items-center space-x-4">
            <Badge variant="outline">
              {progress.completedModules.length} modules completed
            </Badge>
            <Badge variant="secondary">
              {progress.certifications.length} certifications
            </Badge>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="p-6">
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="onboarding">Onboarding</TabsTrigger>
            <TabsTrigger value="modules">Training Modules</TabsTrigger>
            <TabsTrigger value="simulations">Simulations</TabsTrigger>
            <TabsTrigger value="progress">Progress</TabsTrigger>
          </TabsList>

          {/* Onboarding Tab */}
          <TabsContent value="onboarding" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Welcome to the Medical AI Assistant</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p>
                  This comprehensive training program will guide you through:
                </p>
                <ul className="list-disc list-inside space-y-2">
                  <li>Patient safety protocols and emergency procedures</li>
                  <li>Clinical workflow optimization</li>
                  <li>Medical documentation standards</li>
                  <li>Emergency response protocols</li>
                  <li>Communication best practices</li>
                </ul>
                
                <div className="flex space-x-4 pt-4">
                  <Button onClick={() => setActiveTab('modules')}>
                    Start Training
                  </Button>
                  <Button variant="outline">
                    Skip for Now
                  </Button>
                </div>
              </CardContent>
            </Card>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Safety First</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-600">
                    Every module includes critical safety protocols and emergency procedures.
                  </p>
                  <Badge variant="destructive" className="mt-2">
                    ⚠️ Safety Critical
                  </Badge>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Hands-On Learning</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-600">
                    Practice real scenarios through interactive simulations and assessments.
                  </p>
                  <Badge variant="secondary" className="mt-2">
                    🎯 Interactive
                  </Badge>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Training Modules Tab */}
          <TabsContent value="modules" className="space-y-6">
            <div className="grid gap-6">
              {trainingModules.map(module => {
                const isCompleted = progress.completedModules.includes(module.id);
                const score = progress.moduleScores[module.id];
                const attempts = progress.attempts[module.id] || 0;
                const maxAttempts = module.assessment.maxAttempts;

                return (
                  <Card key={module.id} className={isCompleted ? 'border-green-200 bg-green-50' : ''}>
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <CardTitle className="flex items-center space-x-2">
                          <span>{module.title}</span>
                          {isCompleted && <Badge variant="default">✅ Completed</Badge>}
                        </CardTitle>
                        <div className="flex items-center space-x-2">
                          <Badge variant="outline">{module.difficulty}</Badge>
                          <Badge variant="secondary">{module.duration} min</Badge>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <p className="text-gray-600">{module.description}</p>
                      
                      {score !== undefined && (
                        <div className="flex items-center space-x-4">
                          <span>Last Score: {Math.round(score)}%</span>
                          {score >= module.assessment.passingScore ? (
                            <Badge variant="default">Passed</Badge>
                          ) : (
                            <Badge variant="destructive">Failed</Badge>
                          )}
                          <span className="text-sm text-gray-500">
                            Attempt {attempts}/{maxAttempts}
                          </span>
                        </div>
                      )}
                      
                      <div className="flex space-x-2">
                        {isCompleted ? (
                          <Button variant="outline" onClick={() => startAssessment(module.id)}>
                            Review Module
                          </Button>
                        ) : (
                          <Button 
                            onClick={() => startAssessment(module.id)}
                            disabled={attempts >= maxAttempts}
                          >
                            {attempts > 0 ? 'Retake Assessment' : 'Start Module'}
                          </Button>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </TabsContent>

          {/* Simulations Tab */}
          <TabsContent value="simulations" className="space-y-6">
            <div className="space-y-6">
              {medicalWorkflows.map(workflow => (
                <WorkflowSimulation
                  key={workflow.id}
                  workflow={workflow}
                  onStepComplete={(stepId, passed) => {
                    console.log('Step completed:', stepId, passed);
                  }}
                  onWorkflowComplete={(passed) => {
                    console.log('Workflow completed:', passed);
                    if (passed) {
                      // Award certification
                      const newProgress = {
                        ...progress,
                        certifications: [...progress.certifications, workflow.id]
                      };
                      setProgress(newProgress);
                      onProgressUpdate(newProgress);
                    }
                  }}
                />
              ))}
            </div>
          </TabsContent>

          {/* Progress Tab */}
          <TabsContent value="progress" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Training Progress</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Overall Completion</span>
                      <span>{Math.round((progress.completedModules.length / trainingModules.length) * 100)}%</span>
                    </div>
                    <Progress 
                      value={(progress.completedModules.length / trainingModules.length) * 100}
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span>Completed Modules:</span>
                      <Badge variant="default">{progress.completedModules.length}</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Certifications Earned:</span>
                      <Badge variant="secondary">{progress.certifications.length}</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Last Access:</span>
                      <span className="text-sm text-gray-600">
                        {progress.lastAccess.toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Upcoming Requirements</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {trainingModules
                      .filter(module => !progress.completedModules.includes(module.id))
                      .map(module => (
                        <div key={module.id} className="flex items-center justify-between">
                          <span className="text-sm">{module.title}</span>
                          <Badge variant="outline">{module.duration} min</Badge>
                        </div>
                      ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>

        {/* Assessment Modal */}
        {currentAssessment && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <Card className="w-full max-w-2xl max-h-[80vh] overflow-auto">
              <CardHeader>
                <CardTitle>
                  Assessment: {trainingModules.find(m => m.id === currentAssessment.moduleId)?.title}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {trainingModules
                  .find(m => m.id === currentAssessment.moduleId)
                  ?.assessment.questions.map((question, index) => (
                    <div key={question.id} className="border rounded-lg p-4">
                      <h3 className="font-semibold mb-3">
                        {index + 1}. {question.question}
                      </h3>
                      
                      {question.type === 'multiple-choice' && question.options && (
                        <div className="space-y-2">
                          {question.options.map(option => (
                            <label key={option} className="flex items-center space-x-2">
                              <input
                                type="radio"
                                name={question.id}
                                value={option}
                                onChange={(e) => submitAnswer(question.id, e.target.value)}
                                checked={currentAssessment.answers[question.id] === option}
                              />
                              <span>{option}</span>
                            </label>
                          ))}
                        </div>
                      )}
                      
                      {question.type === 'true-false' && (
                        <div className="space-y-2">
                          {['true', 'false'].map(option => (
                            <label key={option} className="flex items-center space-x-2">
                              <input
                                type="radio"
                                name={question.id}
                                value={option}
                                onChange={(e) => submitAnswer(question.id, e.target.value)}
                                checked={currentAssessment.answers[question.id] === option}
                              />
                              <span>{option}</span>
                            </label>
                          ))}
                        </div>
                      )}
                      
                      {question.type === 'scenario' && (
                        <textarea
                          className="w-full border rounded p-2"
                          rows={4}
                          onChange={(e) => submitAnswer(question.id, e.target.value)}
                          value={currentAssessment.answers[question.id] || ''}
                          placeholder="Describe your approach to this scenario..."
                        />
                      )}
                    </div>
                  ))}
                
                <div className="flex justify-end space-x-2">
                  <Button
                    variant="outline"
                    onClick={() => setCurrentAssessment(null)}
                  >
                    Cancel
                  </Button>
                  <Button onClick={completeAssessment}>
                    Submit Assessment
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
};

export default UserOnboardingTraining;