# Contribution Guidelines

Thank you for your interest in contributing to the Medical AI Assistant project! This document provides comprehensive guidelines for contributing to our healthcare AI project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Development Setup](#development-setup)
3. [Coding Standards](#coding-standards)
4. [Testing Requirements](#testing-requirements)
5. [Submission Process](#submission-process)
6. [Review Process](#review-process)
7. [Medical Content Guidelines](#medical-content-guidelines)
8. [Security and Compliance](#security-and-compliance)
9. [Recognition](#recognition)

---

## Code of Conduct

### Our Pledge

We as members, contributors, and leaders pledge to make participation in our community a harassment-free experience for everyone, regardless of age, body size, visible or invisible disability, ethnicity, sex characteristics, gender identity and expression, level of experience, education, socio-economic status, nationality, personal appearance, race, religion, or sexual identity and orientation.

We pledge to act and interact in ways that contribute to an open, welcoming, diverse, inclusive, and healthy community.

### Positive Behavior

Examples of behavior that contributes to a positive environment:

* Demonstrating empathy and kindness toward other people
* Being respectful of differing opinions, viewpoints, and experiences
* Giving and gracefully accepting constructive feedback
* Accepting responsibility and apologizing to those affected by our mistakes, and learning from the experience
* Focusing on what is best not just for us as individuals, but for the overall community
* Using welcoming and inclusive language
* Being patient with newcomers and those learning
* Providing mentorship and support where appropriate

### Healthcare-Specific Standards

Given this project's healthcare focus, we additionally expect:

* **Patient Safety First**: Always prioritize patient safety in discussions and decisions
* **PHI Protection**: Never share, request, or include Protected Health Information (PHI) or any identifiable patient data
* **Evidence-Based**: Base medical/clinical suggestions on evidence and established guidelines
* **Professional Conduct**: Maintain the same professional standards expected in healthcare settings
* **Respectful of Clinical Expertise**: Value contributions from both technical and clinical perspectives
* **Ethical Considerations**: Always consider the ethical implications of AI in healthcare

### Unacceptable Behavior

Examples of unacceptable behavior include:

* The use of sexualized language or imagery, and sexual attention or advances of any kind
* Trolling, insulting or derogatory comments, and personal or political attacks
* Public or private harassment
* Publishing others' private information, such as a physical or email address, without their explicit permission
* Other conduct which could reasonably be considered inappropriate in a professional setting
* Advocating for, or encouraging, any of the above behavior

### Healthcare-Specific Violations

The following violations may result in immediate removal from the project:

* Sharing or requesting PHI or identifiable patient data
* Deliberately introducing unsafe medical logic or advice
* Bypassing or disabling safety features without approval
* Misrepresenting clinical credentials or expertise
* Using the project for unauthorized clinical deployment

### Reporting Violations

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported to:

**clinic-ai-team@yourclinic.example**

All complaints will be reviewed and investigated promptly and fairly. All project maintainers are obligated to respect the privacy and security of the reporter of any incident.

---

## Development Setup

### Prerequisites

Before contributing, ensure you have:

- **Python 3.11+** (backend development)
- **Node.js 22+** (frontend development)
- **Docker & Docker Compose** (containerization)
- **Git** (version control)
- **Basic understanding of healthcare AI ethics and PHI handling**

### Initial Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/Medical-AI-Assistant.git
   cd Medical-AI-Assistant
   ```

2. **Add upstream remote**
   ```bash
   git remote add upstream https://github.com/original/Medical-AI-Assistant.git
   ```

3. **Copy environment template**
   ```bash
   cp .env.example .env
   # Edit .env with your local settings
   ```

4. **Start development environment**
   ```bash
   docker compose -f docker/docker-compose.dev.yml up --build
   ```

5. **Install pre-commit hooks**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

### Development Environment Verification

Verify your setup by running:

```bash
# Backend tests
cd backend && pytest tests/ --cov=.

# Frontend tests
cd frontend && npm test

# E2E tests
npm run test:e2e
```

---

## Coding Standards

### Python (Backend)

**Style Guidelines:**
- Follow **PEP 8** and use Black formatter
- **Line length**: 88 characters (Black default)
- **Type hints**: Required for all function signatures
- **Docstrings**: Use Google-style docstrings

**Example:**
```python
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class PatientQuery:
    """Represents a patient query for processing."""
    query: str
    context: Optional[List[str]] = None
    timestamp: Optional[str] = None

def process_patient_query(
    query: PatientQuery,
    context: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Process a patient query and generate response.
    
    Args:
        query: The patient query object containing the question and metadata
        context: Optional list of conversation history
        
    Returns:
        Dict containing response, safety flags, and metadata
        
    Raises:
        ValueError: If query content is empty or contains PHI
        
    Example:
        >>> query = PatientQuery("I have chest pain", context=["I feel dizzy"])
        >>> result = process_patient_query(query)
        >>> print(result['response'])
        "I understand you're experiencing chest pain..."
    """
    if not query.query.strip():
        raise ValueError("Query cannot be empty")
    
    # Implementation here
    return {"response": "processed", "safety_flags": []}
```

**Linting Requirements:**
- Pass flake8 checks (max line length 88, no unused imports)
- Pass mypy type checking (strict mode enabled)
- Use pydocstyle for docstring consistency

### TypeScript/React (Frontend)

**Style Guidelines:**
- Use Prettier with default settings
- **Type safety**: Strict TypeScript mode enabled
- **Line length**: 100 characters
- **Components**: Functional components with hooks only

**Example:**
```typescript
interface MessageBubbleProps {
  message: string;
  sender: 'patient' | 'assistant';
  timestamp: Date;
  isRedFlag?: boolean;
  safetyScore?: number;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  sender,
  timestamp,
  isRedFlag = false,
  safetyScore
}) => {
  const bubbleClass = [
    'message-bubble',
    `message-bubble--${sender}`,
    isRedFlag ? 'message-bubble--flagged' : '',
  ].join(' ');

  return (
    <div className={bubbleClass} data-testid={`message-${sender}`}>
      <div className="message-content">{message}</div>
      <div className="message-meta">
        <span className="message-timestamp">
          {timestamp.toLocaleTimeString()}
        </span>
        {safetyScore !== undefined && (
          <span className="safety-score">
            Safety: {safetyScore}%
          </span>
        )}
      </div>
    </div>
  );
};
```

### General Principles

**Architecture:**
- **DRY**: Don't Repeat Yourself
- **SOLID**: Follow SOLID principles
- **Clean Architecture**: Separate concerns clearly
- **Security First**: Always validate inputs, sanitize outputs

**Performance:**
- Consider async/await for I/O operations
- Use appropriate caching strategies
- Optimize database queries (avoid N+1 problems)
- Implement pagination for large datasets

**Error Handling:**
- Comprehensive try-catch with proper logging
- Use custom exceptions for domain-specific errors
- Provide meaningful error messages
- Log security-relevant events

---

## Testing Requirements

### Coverage Requirements

**Minimum Coverage:**
- **Unit tests**: 80% code coverage minimum
- **Integration tests**: All API endpoints must have integration tests
- **E2E tests**: Critical user flows (chat, triage, safety checks)

### Running Tests

```bash
# Backend tests with coverage
cd backend
pytest tests/ --cov=. --cov-report=html --cov-report=term

# Frontend tests with coverage
cd frontend
npm test -- --coverage --watchAll=false

# E2E tests
npm run test:e2e

# Security tests
npm run test:security

# All tests
npm run test:all
```

### Writing Tests

**Best Practices:**
- Use **Arrange-Act-Assert** pattern
- Write descriptive test names: `test_safety_filter_blocks_prescriptive_language`
- Mock external dependencies (EHR, model inference, databases)
- Test edge cases: empty inputs, invalid data, timeout scenarios
- Include test data that resembles real medical scenarios (de-identified)

**Example Backend Test:**
```python
import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.safety.filter import SafetyFilter
from src.models.patient import PatientQuery

class TestSafetyFilter:
    """Test cases for the safety filter system."""
    
    def test_safety_filter_blocks_diagnosis_language(self):
        """Test that safety filter blocks diagnostic statements."""
        # Arrange
        filter = SafetyFilter(strictness="high")
        query = PatientQuery(
            query="You have pneumonia. Take antibiotics immediately.",
            timestamp=datetime.now().isoformat()
        )
        
        # Act
        result = filter.check(query.query)
        
        # Assert
        assert result.is_safe == False
        assert "diagnostic language" in result.reason.lower()
        assert result.severity in ["high", "critical"]
        assert result.action == "block"
    
    def test_safety_filter_allows_general_health_info(self):
        """Test that general health information is allowed."""
        # Arrange
        filter = SafetyFilter(strictness="standard")
        query = PatientQuery(
            query="What are the symptoms of common cold?",
            timestamp=datetime.now().isoformat()
        )
        
        # Act
        result = filter.check(query.query)
        
        # Assert
        assert result.is_safe == True
        assert result.severity is None
    
    @patch('src.external.ehr_service.EMRService.get_patient_history')
    def test_integration_with_ehr_system(self, mock_ehr):
        """Test integration with EHR system."""
        # Arrange
        mock_ehr.return_value = {"allergies": ["penicillin"]}
        service = PatientService()
        
        # Act
        result = service.process_patient("123", "What should I avoid?")
        
        # Assert
        assert "penicillin" in result.response.lower()
        mock_ehr.assert_called_once_with("123")
```

**Example Frontend Test:**
```typescript
import { render, screen, fireEvent } from '@testing-library/react';
import { MessageBubble } from '../components/MessageBubble';

describe('MessageBubble', () => {
  it('renders patient message correctly', () => {
    // Arrange
    const timestamp = new Date('2024-01-01T10:00:00');
    
    // Act
    render(
      <MessageBubble
        message="I have a headache"
        sender="patient"
        timestamp={timestamp}
      />
    );
    
    // Assert
    expect(screen.getByText('I have a headache')).toBeInTheDocument();
    expect(screen.getByText('10:00:00 AM')).toBeInTheDocument();
    expect(screen.getByTestId('message-patient')).toHaveClass(
      'message-bubble--patient'
    );
  });
  
  it('displays safety score when provided', () => {
    // Arrange
    const timestamp = new Date();
    
    // Act
    render(
      <MessageBubble
        message="Chest pain"
        sender="patient"
        timestamp={timestamp}
        safetyScore={45}
      />
    );
    
    // Assert
    expect(screen.getByText('Safety: 45%')).toBeInTheDocument();
  });
});
```

### Test Categories

**Required Test Categories:**
1. **Unit Tests**: Individual function/component testing
2. **Integration Tests**: API endpoint testing with real dependencies
3. **Security Tests**: PHI handling, input validation, authentication
4. **Safety Tests**: Medical content safety, red flag detection
5. **Performance Tests**: Response time, concurrent users
6. **E2E Tests**: Complete user workflows

---

## Submission Process

### Branch Naming Convention

Use descriptive branch names with prefixes:
- `feat/` - New features (e.g., `feat/add-red-flag-detection`)
- `fix/` - Bug fixes (e.g., `fix/websocket-reconnection-issue`)
- `docs/` - Documentation updates (e.g., `docs/update-api-reference`)
- `refactor/` - Code refactoring (e.g., `refactor/safety-filter-logic`)
- `test/` - Test additions/updates (e.g., `test/add-coverage-for-triage`)
- `chore/` - Maintenance tasks (e.g., `chore/update-dependencies`)

### Commit Message Format

Follow conventional commits specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `security`

**Example:**
```bash
feat(agent): add safety filter callback for prescriptive language

- Implement regex-based filter for diagnosis keywords
- Add logging for filtered responses
- Include unit tests for edge cases
- Update documentation for new safety features

Closes #123

Breaking Changes:
- Safety filter is now enabled by default (cannot be disabled)
- API response format includes safety metadata

Refs: #456, #789
```

### Pre-Submission Checklist

Before submitting a Pull Request, ensure:

- [ ] Code follows style guidelines (run `black`, `flake8`, `prettier`)
- [ ] All tests pass locally (`pytest`, `npm test`)
- [ ] New tests added for new features (80% coverage minimum)
- [ ] Documentation updated (README, API docs, inline comments)
- [ ] No PHI or secrets in code (scan with `detect-secrets`)
- [ ] Commit messages follow conventional format
- [ ] Branch is up-to-date with main (`git rebase main`)
- [ ] Security scan passes (`npm audit`, `safety check`)
- [ ] Accessibility standards met (WCAG 2.1 AA for frontend)

### Pull Request Template

```markdown
## Description
Brief description of changes and motivation

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that causes existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Security patch
- [ ] Performance improvement

## Testing
- [ ] Unit tests pass (required)
- [ ] Integration tests pass (required)
- [ ] E2E tests pass (for frontend changes)
- [ ] Manual testing performed
- [ ] Security tests pass

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Code is well-commented, particularly in complex areas
- [ ] Documentation updated
- [ ] No new warnings introduced
- [ ] All dependencies are at compatible versions
- [ ] Breaking changes documented in CHANGELOG.md

## Medical Content Review
- [ ] Medical content reviewed by clinical advisor (if applicable)
- [ ] Safety implications considered
- [ ] Evidence-based medical information (if applicable)
- [ ] HIPAA/PHI considerations addressed

## Security & Compliance
- [ ] No PHI exposure risks
- [ ] Input validation implemented
- [ ] Output sanitization verified
- [ ] Authentication/authorization tested
- [ ] Data encryption reviewed

## Screenshots/Evidence
(If applicable, add screenshots, performance graphs, etc.)

## Related Issues
Closes #(issue_number)
Fixes #(issue_number)
Related to #(issue_number)

## Reviewers
- [ ] @maintainer1 (required for medical logic)
- [ ] @maintainer2 (required for security)
- [ ] @clinical-advisor (required for medical content)
```

---

## Review Process

### Review Timeline

- **Initial review**: Within 48 hours (business days)
- **Follow-up reviews**: Within 24 hours after changes
- **Urgent security fixes**: Same day
- **Medical content review**: Within 72 hours

### Reviewer Responsibilities

**What Reviewers Look For:**

1. **Correctness**: Does the code work as intended?
2. **Security**: Any PHI leaks or vulnerabilities?
3. **Safety**: Medical AI safety considerations addressed?
4. **Performance**: Any performance implications?
5. **Maintainability**: Is code clean and well-documented?
6. **Tests**: Adequate test coverage and quality?
7. **Medical Accuracy**: For medical content changes

### Approval Requirements

**Minimum Requirements:**
- 1 maintainer approval required
- All CI checks must pass
- No unresolved conversations
- Up-to-date with target branch
- All automated tests passing

**Additional Requirements for Medical Changes:**
- Clinical advisor approval required
- Safety review completed
- Documentation of medical reasoning
- Evidence-based medical information verified

### Review Process Steps

1. **Automated Checks**: CI/CD pipeline runs
2. **Code Review**: Maintainer reviews code quality
3. **Security Review**: Security-focused review for PHI handling
4. **Medical Review**: Clinical advisor reviews medical content (if applicable)
5. **Final Approval**: Lead maintainer gives final approval
6. **Merge**: Changes are merged to target branch

### Review Comments Guidelines

**For Reviewers:**
- Be specific and actionable in feedback
- Explain the reasoning behind suggestions
- Distinguish between must-fix and nice-to-have changes
- Acknowledge good practices and improvements
- Ask questions to understand intent

**For Contributors:**
- Respond to all review comments
- Explain reasoning if you disagree
- Ask for clarification if feedback is unclear
- Make requested changes promptly
- Keep discussions focused and professional

---

## Medical Content Guidelines

### Clinical Validation Requirements

**For Medical Logic Changes:**

1. **Evidence-Based**: Base medical/clinical suggestions on established guidelines
2. **Clinical Review**: Changes must be reviewed by a clinical advisor
3. **Safety Testing**: Extensive testing with edge cases and failure modes
4. **Documentation**: Clear explanation of medical reasoning and sources

**Sources for Medical Information:**
- Clinical practice guidelines (AMA, WHO, specialty societies)
- Peer-reviewed medical literature
- Medical reference texts
- Institutional protocols

### Safety-Critical Areas

**Extra Scrutiny Required For:**
- Safety filter logic modifications
- Red flag detection algorithms
- PAR (Preliminary Assessment Report) generation
- EHR data handling and privacy
- Consent workflow logic
- Drug interaction checking
- Emergency triage protocols

### Medical Content Standards

**Allowed:**
- General health information and education
- Symptom explanations and descriptions
- Non-prescriptive health guidance
- Red flag identification (urgent care warnings)
- Lifestyle and wellness recommendations

**Prohibited:**
- Specific medical diagnoses
- Prescriptive treatment recommendations
- Medication dosages or administration
- Replacement of professional medical advice
- Emergency medical decisions

### Example: Safe Medical Content

```python
# GOOD: Educational content with proper disclaimers
def generate_symptom_explanation(symptom: str) -> str:
    """Generate educational information about symptoms."""
    if symptom in EMERGENCY_SYMPTOMS:
        return (
            "This symptom may require immediate medical attention. "
            "Please seek emergency care or contact emergency services. "
            "This is general information only and not a substitute for "
            "professional medical advice."
        )
    
    # Generate general educational content
    return f"Common causes of {symptom} include..."

# BAD: Prescriptive medical advice
def generate_treatment_plan(symptoms: List[str]) -> str:
    # NEVER do this
    return f"Based on your symptoms {symptoms}, you should take "
    "amoxicillin 500mg three times daily for 7 days."
```

### Medical Documentation Requirements

**For Every Medical Change:**

1. **Medical Rationale**: Clear explanation of clinical reasoning
2. **Source Citations**: References to medical literature or guidelines
3. **Risk Assessment**: Analysis of potential risks and mitigations
4. **Testing Strategy**: How the change was tested for safety
5. **Clinical Validation**: Evidence that the change improves patient outcomes

---

## Security and Compliance

### PHI Protection Standards

**Never Include:**
- Patient names, addresses, phone numbers
- Medical record numbers (MRN)
- Dates of birth or other identifying information
- Specific medical data that could identify individuals
- Any other Protected Health Information (PHI) under HIPAA

**Always Use:**
- Synthetic or anonymized test data
- De-identified patient information
- Aggregate statistics when possible
- Synthetic medical scenarios for examples

### Security Review Requirements

**Security-Critical Changes Require:**
- Security-focused code review
- Input validation testing
- Output sanitization verification
- Authentication/authorization testing
- Data encryption review

**Required Security Tools:**
```bash
# Install security scanning tools
pip install bandit safety detect-secrets

# Run security scans before submission
bandit -r backend/ -f json -o bandit-report.json
safety check --json > safety-report.json
detect-secrets scan --all-files > secrets-scan.json
```

### Compliance Checklist

**HIPAA Compliance:**
- [ ] No PHI in code or tests
- [ ] Data de-identification verified
- [ ] Access controls implemented
- [ ] Audit logging in place
- [ ] Data encryption verified

**Security Best Practices:**
- [ ] Input validation on all endpoints
- [ ] Output encoding to prevent XSS
- [ ] SQL injection prevention (parameterized queries)
- [ ] Authentication for sensitive operations
- [ ] Authorization checks for all access levels

### Incident Response

**If PHI or Security Issues Are Discovered:**

1. **Immediate Actions:**
   - Stop all development and testing
   - Don't commit or push affected code
   - Notify maintainers immediately
   - Document the issue details

2. **Reporting:**
   - Email: security@yourclinic.example
   - Include: Code location, potential exposure, affected data types
   - Response time: Within 2 hours

3. **Remediation:**
   - Remove all affected code and data
   - Review all related changes
   - Update security procedures if needed
   - Document lessons learned

---

## Recognition

### Contributor Acknowledgment

Contributors will be recognized in several ways:

1. **CONTRIBUTORS.md**: All contributors listed with their contributions
2. **Release Notes**: Significant contributions highlighted in release notes
3. **Documentation**: Contributors credited in relevant documentation
4. **Healthcare Impact**: Recognition of contributions that improve patient safety

### Contribution Categories

**Technical Contributions:**
- Code development and improvements
- Bug fixes and optimizations
- Architecture and design contributions

**Clinical Contributions:**
- Medical content validation
- Safety feature development
- Clinical workflow improvements
- Evidence-based guideline implementation

**Community Contributions:**
- Documentation improvements
- Community support and mentoring
- Testing and quality assurance
- Issue triage and management

### Hall of Fame

**Criteria for Special Recognition:**
- Consistent high-quality contributions over 6+ months
- Critical bug fixes or security patches
- Major feature development or architecture improvements
- Outstanding community support and mentoring
- Medical content expertise that improves patient safety

### Annual Recognition

**Healthcare AI Excellence Awards:**
- **Patient Safety Champion**: Contributions that directly improve patient safety
- **Technical Excellence**: Outstanding technical contributions
- **Community Leadership**: Exceptional community support and mentorship
- **Medical Accuracy**: Contributions that improve medical content quality

---

## Getting Help

### Support Channels

- **GitHub Discussions**: General questions and community discussion
- **GitHub Issues**: Bug reports and feature requests
- **Email**: clinic-ai-team@yourclinic.example (general inquiries)
- **Security Issues**: security@yourclinic.example (security matters only)

### Development Questions

Before asking questions, please:
1. Check existing documentation
2. Search existing issues and discussions
3. Review similar changes in the codebase
4. Test in a clean development environment

### Medical Questions

For medical content questions:
- Consult medical guidelines and literature
- Discuss with clinical advisors
- Propose changes with medical rationale
- Include evidence and source citations

---

**Remember**: This project involves healthcare AI. Patient safety and privacy are paramount. When in doubt, err on the side of caution.

Thank you for contributing to making healthcare AI safer and more accessible for everyone!