import React, { useState, useEffect } from 'react';
import { FaGraduationCap, FaPlay, FaCertificate, FaClock, FaUserMd, FaCheckCircle, FaLock } from 'react-icons/fa';

const Training = () => {
  const [courses, setCourses] = useState([]);
  const [enrollments, setEnrollments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState('');
  const [selectedDifficulty, setSelectedDifficulty] = useState('');

  const categories = [
    { value: '', label: 'All Categories' },
    { value: 'onboarding', label: 'Onboarding' },
    { value: 'advanced', label: 'Advanced Features' },
    { value: 'certification', label: 'Certification' },
    { value: 'compliance', label: 'Compliance & Security' },
    { value: 'medical_workflows', label: 'Medical Workflows' }
  ];

  const difficulties = [
    { value: '', label: 'All Levels' },
    { value: 'beginner', label: 'Beginner' },
    { value: 'intermediate', label: 'Intermediate' },
    { value: 'advanced', label: 'Advanced' }
  ];

  useEffect(() => {
    fetchCourses();
    fetchEnrollments();
  }, [selectedCategory, selectedDifficulty]);

  const fetchCourses = async () => {
    try {
      const queryParams = new URLSearchParams({
        category: selectedCategory,
        difficulty_level: selectedDifficulty
      });

      const response = await fetch(`/api/training/courses?${queryParams}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setCourses(data);
      }
    } catch (error) {
      console.error('Failed to fetch courses:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchEnrollments = async () => {
    try {
      const response = await fetch('/api/training/enrollments', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setEnrollments(data);
      }
    } catch (error) {
      console.error('Failed to fetch enrollments:', error);
    }
  };

  const handleEnroll = async (courseId) => {
    try {
      const response = await fetch(`/api/training/enroll/${courseId}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      if (response.ok) {
        fetchEnrollments();
        fetchCourses();
      }
    } catch (error) {
      console.error('Failed to enroll:', error);
    }
  };

  const getEnrollmentStatus = (courseId) => {
    return enrollments.find(enrollment => enrollment.course_id === courseId);
  };

  const formatDuration = (minutes) => {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    
    if (hours > 0) {
      return `${hours}h ${mins}m`;
    }
    return `${mins}m`;
  };

  const getDifficultyColor = (difficulty) => {
    switch (difficulty) {
      case 'beginner': return '#27ae60';
      case 'intermediate': return '#f39c12';
      case 'advanced': return '#e74c3c';
      default: return '#95a5a6';
    }
  };

  return (
    <div className="training-container">
      <div className="page-header">
        <h1>
          <FaGraduationCap />
          Training & Certification
        </h1>
        <p>Professional development and certification programs for healthcare professionals</p>
      </div>

      {/* Progress Overview */}
      <div className="progress-overview">
        <h2>Your Learning Progress</h2>
        <div className="progress-cards">
          <div className="progress-card">
            <div className="progress-icon">
              <FaGraduationCap />
            </div>
            <div className="progress-content">
              <h3>{enrollments.length}</h3>
              <p>Enrolled Courses</p>
            </div>
          </div>
          
          <div className="progress-card">
            <div className="progress-icon">
              <FaCheckCircle />
            </div>
            <div className="progress-content">
              <h3>{enrollments.filter(e => e.passed).length}</h3>
              <p>Completed Courses</p>
            </div>
          </div>
          
          <div className="progress-card">
            <div className="progress-icon">
              <FaCertificate />
            </div>
            <div className="progress-content">
              <h3>{enrollments.filter(e => e.certificate_issued).length}</h3>
              <p>Certificates Earned</p>
            </div>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="training-filters">
        <select
          value={selectedCategory}
          onChange={(e) => setSelectedCategory(e.target.value)}
        >
          {categories.map(cat => (
            <option key={cat.value} value={cat.value}>
              {cat.label}
            </option>
          ))}
        </select>

        <select
          value={selectedDifficulty}
          onChange={(e) => setSelectedDifficulty(e.target.value)}
        >
          {difficulties.map(diff => (
            <option key={diff.value} value={diff.value}>
              {diff.label}
            </option>
          ))}
        </select>
      </div>

      {/* Available Courses */}
      <div className="courses-section">
        <h2>Available Courses</h2>
        <div className="courses-grid">
          {loading ? (
            <div className="loading">Loading courses...</div>
          ) : courses.length === 0 ? (
            <div className="no-courses">
              <FaGraduationCap />
              <h3>No courses found</h3>
              <p>Try adjusting your filter criteria.</p>
            </div>
          ) : (
            courses.map(course => {
              const enrollment = getEnrollmentStatus(course.id);
              
              return (
                <div key={course.id} className={`course-card ${enrollment ? 'enrolled' : ''}`}>
                  <div className="course-header">
                    <div className="course-meta">
                      <span 
                        className="difficulty-badge"
                        style={{ backgroundColor: getDifficultyColor(course.difficulty_level) }}
                      >
                        {course.difficulty_level}
                      </span>
                      {course.is_medical_training && (
                        <FaUserMd className="medical-icon" title="Medical Training" />
                      )}
                      {course.is_certification_required && (
                        <span className="certification-badge">
                          <FaCertificate />
                          Certification
                        </span>
                      )}
                    </div>
                    <h3>{course.title}</h3>
                  </div>

                  <div className="course-info">
                    <div className="course-stats">
                      <div className="stat">
                        <FaClock />
                        <span>{formatDuration(course.duration_minutes)}</span>
                      </div>
                      <div className="stat">
                        <span>Category: {course.category.replace('_', ' ')}</span>
                      </div>
                    </div>
                    
                    {enrollment && (
                      <div className="enrollment-status">
                        {enrollment.passed ? (
                          <div className="status-completed">
                            <FaCheckCircle />
                            Completed
                          </div>
                        ) : enrollment.completion_date ? (
                          <div className="status-pending">
                            Pending Review
                          </div>
                        ) : (
                          <div className="status-in-progress">
                            In Progress
                          </div>
                        )}
                      </div>
                    )}
                  </div>

                  <div className="course-description">
                    <p>{course.description}</p>
                  </div>

                  {course.learning_objectives && course.learning_objectives.length > 0 && (
                    <div className="learning-objectives">
                      <h4>Learning Objectives:</h4>
                      <ul>
                        {course.learning_objectives.slice(0, 3).map((objective, index) => (
                          <li key={index}>{objective}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  <div className="course-actions">
                    {enrollment ? (
                      <button className="btn-primary">
                        <FaPlay />
                        {enrollment.passed ? 'Retake Course' : 'Continue Learning'}
                      </button>
                    ) : (
                      <button 
                        onClick={() => handleEnroll(course.id)}
                        className="btn-enroll"
                      >
                        Enroll Now
                      </button>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* Current Enrollments */}
      {enrollments.length > 0 && (
        <div className="current-enrollments">
          <h2>Current Enrollments</h2>
          <div className="enrollments-list">
            {enrollments.map(enrollment => {
              const course = courses.find(c => c.id === enrollment.course_id);
              if (!course) return null;

              return (
                <div key={enrollment.id} className="enrollment-item">
                  <div className="enrollment-course">
                    <h4>{course.title}</h4>
                    <span className="category">{course.category.replace('_', ' ')}</span>
                  </div>
                  
                  <div className="enrollment-progress">
                    {enrollment.passed ? (
                      <div className="progress-badge passed">
                        <FaCheckCircle />
                        Passed
                      </div>
                    ) : enrollment.completion_date ? (
                      <div className="progress-badge pending">
                        Under Review
                      </div>
                    ) : (
                      <div className="progress-badge in-progress">
                        In Progress
                      </div>
                    )}
                    
                    {enrollment.score && (
                      <span className="score">Score: {enrollment.score}%</span>
                    )}
                  </div>

                  <div className="enrollment-actions">
                    {enrollment.certificate_issued && (
                      <a 
                        href={enrollment.certificate_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="btn-certificate"
                      >
                        <FaCertificate />
                        View Certificate
                      </a>
                    )}
                    
                    <button className="btn-primary">
                      <FaPlay />
                      {enrollment.passed ? 'Review' : 'Continue'}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Certification Programs */}
      <div className="certification-programs">
        <h2>Professional Certifications</h2>
        <div className="certifications-grid">
          <div className="certification-card">
            <FaUserMd className="cert-icon" />
            <h3>Medical AI Certification</h3>
            <p>Comprehensive certification for healthcare professionals using AI systems</p>
            <div className="cert-details">
              <span>Duration: 8 hours</span>
              <span>Prerequisites: Basic medical knowledge</span>
              <span>Valid: 1 year</span>
            </div>
            <button className="btn-primary">
              Learn More
            </button>
          </div>
          
          <div className="certification-card">
            <FaCertificate className="cert-icon" />
            <h3>Data Security Specialist</h3>
            <p>Advanced training in healthcare data security and HIPAA compliance</p>
            <div className="cert-details">
              <span>Duration: 12 hours</span>
              <span>Prerequisites: Security fundamentals</span>
              <span>Valid: 2 years</span>
            </div>
            <button className="btn-primary">
              Learn More
            </button>
          </div>
          
          <div className="certification-card">
            <FaGraduationCap className="cert-icon" />
            <h3>Platform Administrator</h3>
            <p>Administrator-level training for system management and user support</p>
            <div className="cert-details">
              <span>Duration: 16 hours</span>
              <span>Prerequisites: Technical experience</span>
              <span>Valid: 2 years</span>
            </div>
            <button className="btn-primary">
              Learn More
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Training;