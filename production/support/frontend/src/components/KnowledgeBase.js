import React, { useState, useEffect } from 'react';
import { FaSearch, FaBook, FaFileAlt, FaVideo, FaDownload, FaThumbsUp, FaThumbsDown, FaEye, FaUserMd } from 'react-icons/fa';

const KnowledgeBase = () => {
  const [articles, setArticles] = useState([]);
  const [filteredArticles, setFilteredArticles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedArticle, setSelectedArticle] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('');
  const [selectedDifficulty, setSelectedDifficulty] = useState('');
  const [medicalSpecialty, setMedicalSpecialty] = useState('');

  const categories = [
    { value: '', label: 'All Categories' },
    { value: 'getting_started', label: 'Getting Started' },
    { value: 'medical_features', label: 'Medical Features' },
    { value: 'api_integration', label: 'API & Integration' },
    { value: 'troubleshooting', label: 'Troubleshooting' },
    { value: 'best_practices', label: 'Best Practices' },
    { value: 'compliance', label: 'Compliance & Security' },
    { value: 'training', label: 'Training & Tutorials' }
  ];

  const difficulties = [
    { value: '', label: 'All Levels' },
    { value: 'beginner', label: 'Beginner' },
    { value: 'intermediate', label: 'Intermediate' },
    { value: 'advanced', label: 'Advanced' }
  ];

  const specialties = [
    { value: '', label: 'All Specialties' },
    { value: 'cardiology', label: 'Cardiology' },
    { value: 'radiology', label: 'Radiology' },
    { value: 'pathology', label: 'Pathology' },
    { value: 'emergency_medicine', label: 'Emergency Medicine' },
    { value: 'pediatrics', label: 'Pediatrics' },
    { value: 'surgery', label: 'Surgery' }
  ];

  useEffect(() => {
    fetchArticles();
  }, []);

  useEffect(() => {
    filterArticles();
  }, [articles, searchTerm, selectedCategory, selectedDifficulty, medicalSpecialty]);

  const fetchArticles = async () => {
    try {
      const queryParams = new URLSearchParams({
        category: selectedCategory,
        difficulty_level: selectedDifficulty,
        medical_specialty: medicalSpecialty
      });

      const response = await fetch(`/api/kb/articles?${queryParams}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setArticles(data);
      }
    } catch (error) {
      console.error('Failed to fetch articles:', error);
    } finally {
      setLoading(false);
    }
  };

  const filterArticles = () => {
    let filtered = articles;

    if (searchTerm) {
      filtered = filtered.filter(article =>
        article.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        article.content.toLowerCase().includes(searchTerm.toLowerCase()) ||
        article.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()))
      );
    }

    if (selectedCategory) {
      filtered = filtered.filter(article => article.category === selectedCategory);
    }

    if (selectedDifficulty) {
      filtered = filtered.filter(article => article.difficulty_level === selectedDifficulty);
    }

    if (medicalSpecialty) {
      filtered = filtered.filter(article => article.medical_specialty === medicalSpecialty);
    }

    setFilteredArticles(filtered);
  };

  const handleVote = async (articleId, isHelpful) => {
    try {
      await fetch(`/api/kb/articles/${articleId}/vote`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({ helpful: isHelpful })
      });
      
      fetchArticles(); // Refresh to update vote counts
    } catch (error) {
      console.error('Failed to submit vote:', error);
    }
  };

  const getDifficultyColor = (difficulty) => {
    switch (difficulty) {
      case 'beginner': return '#27ae60';
      case 'intermediate': return '#f39c12';
      case 'advanced': return '#e74c3c';
      default: return '#95a5a6';
    }
  };

  if (selectedArticle) {
    return <ArticleView article={selectedArticle} onBack={() => setSelectedArticle(null)} />;
  }

  return (
    <div className="knowledge-base-container">
      <div className="page-header">
        <h1>
          <FaBook />
          Knowledge Base
        </h1>
        <p>Comprehensive guides and documentation for healthcare professionals</p>
      </div>

      {/* Search and Filters */}
      <div className="kb-filters">
        <div className="search-section">
          <div className="search-box">
            <FaSearch className="search-icon" />
            <input
              type="text"
              placeholder="Search articles, guides, and tutorials..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
        </div>

        <div className="filter-controls">
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

          <select
            value={medicalSpecialty}
            onChange={(e) => setMedicalSpecialty(e.target.value)}
          >
            {specialties.map(spec => (
              <option key={spec.value} value={spec.value}>
                {spec.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Articles Grid */}
      <div className="articles-grid">
        {loading ? (
          <div className="loading">Loading articles...</div>
        ) : filteredArticles.length === 0 ? (
          <div className="no-articles">
            <FaFileAlt />
            <h3>No articles found</h3>
            <p>Try adjusting your search criteria or browse by category.</p>
          </div>
        ) : (
          filteredArticles.map(article => (
            <div key={article.id} className="article-card" onClick={() => setSelectedArticle(article)}>
              <div className="article-header">
                <div className="article-meta">
                  <span 
                    className="difficulty-badge"
                    style={{ backgroundColor: getDifficultyColor(article.difficulty_level) }}
                  >
                    {article.difficulty_level}
                  </span>
                  {article.is_medical_content && (
                    <FaUserMd className="medical-icon" title="Medical Content" />
                  )}
                </div>
                <h3>{article.title}</h3>
              </div>

              <div className="article-summary">
                <p>{article.summary || article.content.substring(0, 150) + '...'}</p>
              </div>

              <div className="article-footer">
                <div className="article-stats">
                  <div className="stat">
                    <FaEye />
                    <span>{article.view_count}</span>
                  </div>
                  <div className="stat">
                    <FaThumbsUp />
                    <span>{article.helpful_votes}</span>
                  </div>
                  <div className="stat">
                    <FaThumbsDown />
                    <span>{article.not_helpful_votes}</span>
                  </div>
                </div>
                <div className="article-category">
                  {article.category.replace('_', ' ')}
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Quick Access Section */}
      <div className="quick-access">
        <h2>Quick Access</h2>
        <div className="quick-links">
          <div className="quick-link">
            <FaFileAlt />
            <h3>Getting Started</h3>
            <p>New to the platform? Start here for onboarding guides</p>
          </div>
          <div className="quick-link">
            <FaVideo />
            <h3>Video Tutorials</h3>
            <p>Watch step-by-step guides for key features</p>
          </div>
          <div className="quick-link">
            <FaUserMd />
            <h3>Medical Guidelines</h3>
            <p>Clinical workflows and medical best practices</p>
          </div>
          <div className="quick-link">
            <FaDownload />
            <h3>API Documentation</h3>
            <p>Developer resources and integration guides</p>
          </div>
        </div>
      </div>
    </div>
  );
};

const ArticleView = ({ article, onBack }) => {
  const [helpful, setHelpful] = useState(null);

  const handleVote = async (isHelpful) => {
    try {
      await fetch(`/api/kb/articles/${article.id}/vote`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({ helpful: isHelpful })
      });
      
      setHelpful(isHelpful);
    } catch (error) {
      console.error('Failed to submit vote:', error);
    }
  };

  return (
    <div className="article-view-container">
      <div className="article-header">
        <button onClick={onBack} className="btn-back">
          ← Back to Knowledge Base
        </button>
        <div className="article-meta">
          <span className="category">{article.category.replace('_', ' ')}</span>
          {article.is_medical_content && (
            <span className="medical-badge">
              <FaUserMd />
              Medical Content
            </span>
          )}
        </div>
        <h1>{article.title}</h1>
        <div className="article-info">
          <span>Last updated: {new Date(article.updated_at).toLocaleDateString()}</span>
          <span>Views: {article.view_count}</span>
        </div>
      </div>

      <div className="article-content">
        <div className="content-body">
          <div dangerouslySetInnerHTML={{ __html: article.content }} />
        </div>

        <div className="article-tags">
          {article.tags.map(tag => (
            <span key={tag} className="tag">#{tag}</span>
          ))}
        </div>
      </div>

      <div className="article-feedback">
        <h3>Was this article helpful?</h3>
        <div className="feedback-buttons">
          <button 
            onClick={() => handleVote(true)}
            className={`feedback-btn helpful ${helpful === true ? 'active' : ''}`}
          >
            <FaThumbsUp />
            Yes ({article.helpful_votes})
          </button>
          <button 
            onClick={() => handleVote(false)}
            className={`feedback-btn not-helpful ${helpful === false ? 'active' : ''}`}
          >
            <FaThumbsDown />
            No ({article.not_helpful_votes})
          </button>
        </div>
      </div>
    </div>
  );
};

export default KnowledgeBase;