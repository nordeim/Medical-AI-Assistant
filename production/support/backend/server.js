const express = require('express');
const { Pool } = require('pg');
const rateLimit = require('express-rate-limit');
const helmet = require('helmet');
const cors = require('cors');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');
const bodyParser = require('body-parser');
const morgan = require('morgan');
require('dotenv').config();

const app = express();
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  max: parseInt(process.env.DB_POOL_MAX || '20'),
  min: parseInt(process.env.DB_POOL_MIN || '5')
});

// Middleware
app.use(helmet());
app.use(cors());
app.use(bodyParser.json({ limit: '10mb' }));
app.use(bodyParser.urlencoded({ extended: true, limit: '10mb' }));
app.use(morgan('combined'));

// Rate limiting
const limiter = rateLimit({
  windowMs: parseInt(process.env.RATE_LIMIT_WINDOW) * 1000,
  max: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS),
  message: 'Too many requests from this IP'
});
app.use('/api/', limiter);

// Authentication middleware
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return res.status(401).json({ error: 'Access token required' });
  }

  jwt.verify(token, process.env.JWT_SECRET, (err, user) => {
    if (err) {
      return res.status(403).json({ error: 'Invalid or expired token' });
    }
    req.user = user;
    next();
  });
};

// Generate ticket number
const generateTicketNumber = () => {
  const prefix = 'HC';
  const timestamp = Date.now().toString().slice(-8);
  const random = Math.random().toString(36).substr(2, 4).toUpperCase();
  return `${prefix}-${timestamp}-${random}`;
};

// Generate incident number
const generateIncidentNumber = () => {
  const prefix = 'INC';
  const timestamp = Date.now().toString().slice(-8);
  const random = Math.random().toString(36).substr(2, 4).toUpperCase();
  return `${prefix}-${timestamp}-${random}`;
};

// Support Ticket Routes
app.post('/api/tickets', authenticateToken, async (req, res) => {
  try {
    const { title, description, category, priority, medical_case_id } = req.body;
    const ticketNumber = generateTicketNumber();
    
    // Medical emergency escalation
    const isMedicalEmergency = category === 'medical' && priority === 'emergency';
    const estimatedResolutionTime = isMedicalEmergency ? 
      `${process.env.MEDICAL_EMERGENCY_SLA} minutes` : 
      '4 hours';

    const query = `
      INSERT INTO support_tickets (
        organization_id, user_id, ticket_number, title, description, 
        category, priority, medical_case_id, estimated_resolution_time
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
      RETURNING *`;
    
    const values = [
      req.user.organization_id,
      req.user.id,
      ticketNumber,
      title,
      description,
      category,
      priority,
      medical_case_id || null,
      estimatedResolutionTime
    ];

    const result = await pool.query(query, values);
    
    // Send notifications for high priority or medical tickets
    if (priority === 'high' || priority === 'critical' || isMedicalEmergency) {
      await sendNotification({
        type: 'ticket_created',
        ticket: result.rows[0],
        priority: isMedicalEmergency ? 'emergency' : priority
      });
    }

    res.status(201).json(result.rows[0]);
  } catch (error) {
    console.error('Error creating ticket:', error);
    res.status(500).json({ error: 'Failed to create ticket' });
  }
});

app.get('/api/tickets', authenticateToken, async (req, res) => {
  try {
    const { status, priority, category, page = 1, limit = 20 } = req.query;
    const offset = (page - 1) * limit;
    
    let query = `
      SELECT st.*, u.full_name as user_name, org.name as organization_name
      FROM support_tickets st
      JOIN users u ON st.user_id = u.id
      JOIN organizations org ON st.organization_id = org.id
      WHERE st.organization_id = $1`;
    
    const values = [req.user.organization_id];
    let paramCount = 1;

    if (status) {
      paramCount++;
      query += ` AND st.status = $${paramCount}`;
      values.push(status);
    }

    if (priority) {
      paramCount++;
      query += ` AND st.priority = $${paramCount}`;
      values.push(priority);
    }

    if (category) {
      paramCount++;
      query += ` AND st.category = $${paramCount}`;
      values.push(category);
    }

    query += ` ORDER BY st.created_at DESC LIMIT $${paramCount + 1} OFFSET $${paramCount + 2}`;
    values.push(limit, offset);

    const result = await pool.query(query, values);
    res.json(result.rows);
  } catch (error) {
    console.error('Error fetching tickets:', error);
    res.status(500).json({ error: 'Failed to fetch tickets' });
  }
});

app.put('/api/tickets/:id', authenticateToken, async (req, res) => {
  try {
    const { status, assigned_to, satisfaction_rating, satisfaction_feedback } = req.body;
    const ticketId = req.params.id;

    const query = `
      UPDATE support_tickets 
      SET status = $1, assigned_to = $2, satisfaction_rating = $3, 
          satisfaction_feedback = $4, updated_at = NOW()
      WHERE id = $5 AND organization_id = $6
      RETURNING *`;
    
    const values = [
      status,
      assigned_to,
      satisfaction_rating || null,
      satisfaction_feedback,
      ticketId,
      req.user.organization_id
    ];

    const result = await pool.query(query, values);

    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'Ticket not found' });
    }

    res.json(result.rows[0]);
  } catch (error) {
    console.error('Error updating ticket:', error);
    res.status(500).json({ error: 'Failed to update ticket' });
  }
});

// User Feedback Routes
app.post('/api/feedback', authenticateToken, async (req, res) => {
  try {
    const { feedback_type, rating, title, description, feature_area, is_anonymous } = req.body;
    
    // Sentiment analysis
    const sentiment_score = await analyzeSentiment(description);

    const query = `
      INSERT INTO user_feedback (
        organization_id, user_id, feedback_type, rating, sentiment_score,
        title, description, feature_area, is_anonymous
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
      RETURNING *`;
    
    const values = [
      req.user.organization_id,
      req.user.id,
      feedback_type,
      rating,
      sentiment_score,
      title,
      description,
      feature_area,
      is_anonymous || false
    ];

    const result = await pool.query(query, values);
    res.status(201).json(result.rows[0]);
  } catch (error) {
    console.error('Error creating feedback:', error);
    res.status(500).json({ error: 'Failed to create feedback' });
  }
});

app.get('/api/feedback', authenticateToken, async (req, res) => {
  try {
    const { feedback_type, feature_area, page = 1, limit = 20 } = req.query;
    const offset = (page - 1) * limit;
    
    let query = `
      SELECT uf.*, u.full_name as user_name
      FROM user_feedback uf
      LEFT JOIN users u ON uf.user_id = u.id
      WHERE uf.organization_id = $1`;
    
    const values = [req.user.organization_id];
    let paramCount = 1;

    if (feedback_type) {
      paramCount++;
      query += ` AND uf.feedback_type = $${paramCount}`;
      values.push(feedback_type);
    }

    if (feature_area) {
      paramCount++;
      query += ` AND uf.feature_area = $${paramCount}`;
      values.push(feature_area);
    }

    query += ` ORDER BY uf.created_at DESC LIMIT $${paramCount + 1} OFFSET $${paramCount + 2}`;
    values.push(limit, offset);

    const result = await pool.query(query, values);
    res.json(result.rows);
  } catch (error) {
    console.error('Error fetching feedback:', error);
    res.status(500).json({ error: 'Failed to fetch feedback' });
  }
});

// Health Check Routes
app.get('/api/health', async (req, res) => {
  try {
    const services = [
      'database',
      'ai_api',
      'notification_service',
      'file_storage'
    ];

    const healthChecks = [];
    
    for (const service of services) {
      const check = await performHealthCheck(service);
      healthChecks.push(check);
    }

    const allHealthy = healthChecks.every(check => check.status === 'healthy');
    
    res.json({
      status: allHealthy ? 'healthy' : 'degraded',
      timestamp: new Date().toISOString(),
      services: healthChecks,
      uptime: process.uptime(),
      memory: process.memoryUsage()
    });
  } catch (error) {
    console.error('Error performing health check:', error);
    res.status(500).json({ 
      status: 'unhealthy', 
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

app.get('/api/health/:service', async (req, res) => {
  try {
    const service = req.params.service;
    const check = await performHealthCheck(service);
    res.json(check);
  } catch (error) {
    console.error(`Error checking service ${service}:`, error);
    res.status(500).json({ 
      status: 'unhealthy', 
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Incident Management Routes
app.post('/api/incidents', authenticateToken, async (req, res) => {
  try {
    const { title, description, severity, impact_description, affected_services, medical_emergency } = req.body;
    const incidentNumber = generateIncidentNumber();

    const query = `
      INSERT INTO incidents (
        incident_number, title, description, severity, status,
        impact_description, affected_services, affected_organizations,
        medical_emergency, reported_by
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
      RETURNING *`;
    
    const values = [
      incidentNumber,
      title,
      description,
      severity,
      'investigating',
      impact_description,
      affected_services,
      [req.user.organization_id],
      medical_emergency || false,
      req.user.id
    ];

    const result = await pool.query(query, values);

    // Medical emergency escalation
    if (medical_emergency) {
      await escalateMedicalEmergency(result.rows[0]);
    }

    res.status(201).json(result.rows[0]);
  } catch (error) {
    console.error('Error creating incident:', error);
    res.status(500).json({ error: 'Failed to create incident' });
  }
});

app.get('/api/incidents', authenticateToken, async (req, res) => {
  try {
    const { status, severity, page = 1, limit = 20 } = req.query;
    const offset = (page - 1) * limit;
    
    let query = `
      SELECT i.*, r.full_name as reported_by_name, a.full_name as assigned_to_name
      FROM incidents i
      LEFT JOIN users r ON i.reported_by = r.id
      LEFT JOIN users a ON i.assigned_to = a.id
      WHERE $1 = ANY(i.affected_organizations)`;
    
    const values = [req.user.organization_id];
    let paramCount = 1;

    if (status) {
      paramCount++;
      query += ` AND i.status = $${paramCount}`;
      values.push(status);
    }

    if (severity) {
      paramCount++;
      query += ` AND i.severity = $${paramCount}`;
      values.push(severity);
    }

    query += ` ORDER BY i.created_at DESC LIMIT $${paramCount + 1} OFFSET $${paramCount + 2}`;
    values.push(limit, offset);

    const result = await pool.query(query, values);
    res.json(result.rows);
  } catch (error) {
    console.error('Error fetching incidents:', error);
    res.status(500).json({ error: 'Failed to fetch incidents' });
  }
});

// Success Metrics Routes
app.get('/api/metrics', authenticateToken, async (req, res) => {
  try {
    const { metric_type, start_date, end_date } = req.query;
    
    let query = `
      SELECT * FROM success_metrics
      WHERE organization_id = $1`;
    
    const values = [req.user.organization_id];
    let paramCount = 1;

    if (metric_type) {
      paramCount++;
      query += ` AND metric_type = $${paramCount}`;
      values.push(metric_type);
    }

    if (start_date) {
      paramCount++;
      query += ` AND measurement_date >= $${paramCount}`;
      values.push(start_date);
    }

    if (end_date) {
      paramCount++;
      query += ` AND measurement_date <= $${paramCount}`;
      values.push(end_date);
    }

    query += ` ORDER BY measurement_date DESC`;

    const result = await pool.query(query, values);
    res.json(result.rows);
  } catch (error) {
    console.error('Error fetching metrics:', error);
    res.status(500).json({ error: 'Failed to fetch metrics' });
  }
});

// Knowledge Base Routes
app.get('/api/kb/articles', async (req, res) => {
  try {
    const { category, search, difficulty_level, medical_specialty, page = 1, limit = 20 } = req.query;
    const offset = (page - 1) * limit;
    
    let query = `
      SELECT * FROM kb_articles
      WHERE is_published = true`;
    
    const values = [];
    let paramCount = 0;

    if (category) {
      paramCount++;
      query += ` AND category = $${paramCount}`;
      values.push(category);
    }

    if (difficulty_level) {
      paramCount++;
      query += ` AND difficulty_level = $${paramCount}`;
      values.push(difficulty_level);
    }

    if (medical_specialty) {
      paramCount++;
      query += ` AND medical_specialty = $${paramCount}`;
      values.push(medical_specialty);
    }

    if (search) {
      paramCount++;
      query += ` AND (title ILIKE $${paramCount} OR content ILIKE $${paramCount})`;
      values.push(`%${search}%`);
    }

    query += ` ORDER BY view_count DESC, created_at DESC LIMIT $${paramCount + 1} OFFSET $${paramCount + 2}`;
    values.push(limit, offset);

    const result = await pool.query(query, values);
    res.json(result.rows);
  } catch (error) {
    console.error('Error fetching KB articles:', error);
    res.status(500).json({ error: 'Failed to fetch knowledge base articles' });
  }
});

// Training System Routes
app.get('/api/training/courses', authenticateToken, async (req, res) => {
  try {
    const { category, difficulty_level, medical_specialty } = req.query;
    
    let query = `
      SELECT * FROM training_courses
      WHERE is_active = true`;
    
    const values = [];
    let paramCount = 0;

    if (category) {
      paramCount++;
      query += ` AND category = $${paramCount}`;
      values.push(category);
    }

    if (difficulty_level) {
      paramCount++;
      query += ` AND difficulty_level = $${paramCount}`;
      values.push(difficulty_level);
    }

    if (medical_specialty) {
      paramCount++;
      query += ` AND medical_specialty = $${paramCount}`;
      values.push(medical_specialty);
    }

    query += ` ORDER BY created_at DESC`;

    const result = await pool.query(query, values);
    res.json(result.rows);
  } catch (error) {
    console.error('Error fetching courses:', error);
    res.status(500).json({ error: 'Failed to fetch courses' });
  }
});

app.post('/api/training/enroll/:courseId', authenticateToken, async (req, res) => {
  try {
    const courseId = req.params.courseId;
    
    const query = `
      INSERT INTO course_enrollments (course_id, user_id)
      VALUES ($1, $2)
      RETURNING *`;
    
    const result = await pool.query(query, [courseId, req.user.id]);
    res.status(201).json(result.rows[0]);
  } catch (error) {
    console.error('Error enrolling in course:', error);
    res.status(500).json({ error: 'Failed to enroll in course' });
  }
});

// Helper functions
async function performHealthCheck(service) {
  const timestamp = new Date().toISOString();
  
  try {
    switch (service) {
      case 'database':
        const dbStart = Date.now();
        await pool.query('SELECT 1');
        const dbResponseTime = Date.now() - dbStart;
        
        return {
          service: 'database',
          status: 'healthy',
          response_time_ms: dbResponseTime,
          checked_at: timestamp
        };
        
      case 'ai_api':
        const aiStart = Date.now();
        const response = await fetch(`${process.env.SENTIMENT_API_URL}/health`);
        const aiResponseTime = Date.now() - aiStart;
        
        return {
          service: 'ai_api',
          status: response.ok ? 'healthy' : 'degraded',
          response_time_ms: aiResponseTime,
          checked_at: timestamp
        };
        
      default:
        return {
          service,
          status: 'unknown',
          error: 'Service not configured',
          checked_at: timestamp
        };
    }
  } catch (error) {
    return {
      service,
      status: 'unhealthy',
      error: error.message,
      checked_at: timestamp
    };
  }
}

async function analyzeSentiment(text) {
  try {
    const response = await fetch(`${process.env.SENTIMENT_API_URL}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.SENTIMENT_API_KEY}`
      },
      body: JSON.stringify({ text })
    });
    
    if (response.ok) {
      const result = await response.json();
      return result.sentiment_score;
    }
  } catch (error) {
    console.error('Sentiment analysis failed:', error);
  }
  return 0; // Neutral sentiment
}

async function sendNotification({ type, ticket, priority }) {
  try {
    // Email notification
    if (process.env.ENABLE_EMAIL_NOTIFICATIONS) {
      await sendEmail({
        to: process.env.SMTP_FROM_EMAIL,
        subject: `New ${priority} ticket: ${ticket.title}`,
        body: `Ticket ${ticket.ticket_number} has been created with ${priority} priority`
      });
    }
    
    // Slack notification for critical issues
    if ((priority === 'critical' || priority === 'emergency') && process.env.SLACK_WEBHOOK_URL) {
      await sendSlackMessage({
        text: `🚨 Critical Support Ticket: ${ticket.title}\nPriority: ${priority}\nTicket: ${ticket.ticket_number}`
      });
    }
  } catch (error) {
    console.error('Notification failed:', error);
  }
}

async function escalateMedicalEmergency(incident) {
  try {
    // Immediate escalation for medical emergencies
    if (process.env.MEDICAL_ESCALATION_EMAIL) {
      await sendEmail({
        to: process.env.MEDICAL_ESCALATION_EMAIL,
        subject: `🚨 MEDICAL EMERGENCY: ${incident.title}`,
        body: `A medical emergency has been reported. Please respond immediately.\n\nIncident: ${incident.incident_number}\nDescription: ${incident.description}`
      });
    }
  } catch (error) {
    console.error('Medical escalation failed:', error);
  }
}

async function sendEmail({ to, subject, body }) {
  // Email sending implementation
  console.log(`Email sent to ${to}: ${subject}`);
}

async function sendSlackMessage({ text }) {
  try {
    const response = await fetch(process.env.SLACK_WEBHOOK_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
  } catch (error) {
    console.error('Slack notification failed:', error);
  }
}

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('Unhandled error:', error);
  res.status(500).json({ error: 'Internal server error' });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({ error: 'Endpoint not found' });
});

// Start server
const PORT = process.env.APP_PORT || 8080;
app.listen(PORT, () => {
  console.log(`Healthcare Support System running on port ${PORT}`);
});

module.exports = app;