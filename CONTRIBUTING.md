# Contributing to Medical AI Assistant

Thank you for your interest in contributing to the Medical AI Assistant project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 22+
- Docker & Docker Compose
- Git
- Basic understanding of healthcare AI ethics and PHI handling

### Setting Up Development Environment

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/Medical-AI-Assistant.git
   cd Medical-AI-Assistant
   ```

2. **Copy environment template**
   ```bash
   cp .env.example .env
   # Edit .env with your local settings
   ```

3. **Start development environment**
   ```bash
   docker compose -f docker/docker-compose.dev.yml up --build
   ```

4. **Install pre-commit hooks**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Development Workflow

### Branch Naming Convention

Use descriptive branch names with prefixes:
- `feat/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions/updates
- `chore/` - Maintenance tasks

Example: `feat/add-safety-filter`, `fix/websocket-reconnection`

### Commit Message Format

Follow conventional commits:
```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example:
```
feat(agent): add safety filter callback for prescriptive language

- Implement regex-based filter for diagnosis keywords
- Add logging for filtered responses
- Include unit tests for edge cases

Closes #123
```

## Coding Standards

### Python (Backend)

- **Style**: Follow PEP 8 and use Black formatter
- **Type hints**: Required for all function signatures
- **Docstrings**: Use Google-style docstrings
- **Linting**: Pass flake8 and mypy checks
- **Line length**: 88 characters (Black default)

Example:
```python
from typing import List, Optional

def process_patient_query(
    query: str,
    context: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Process a patient query and generate response.
    
    Args:
        query: The patient's question or statement
        context: Optional list of conversation history
        
    Returns:
        Dict containing response and metadata
        
    Raises:
        ValueError: If query is empty or invalid
    """
    pass
```

### TypeScript/React (Frontend)

- **Style**: Use Prettier with default settings
- **Type safety**: Strict TypeScript mode enabled
- **Components**: Functional components with hooks
- **Props**: Define explicit interfaces for all props
- **Line length**: 100 characters

Example:
```typescript
interface MessageBubbleProps {
  message: string;
  sender: 'patient' | 'assistant';
  timestamp: Date;
  isRedFlag?: boolean;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  sender,
  timestamp,
  isRedFlag = false
}) => {
  // Implementation
};
```

### General Principles

- **DRY**: Don't Repeat Yourself
- **SOLID**: Follow SOLID principles
- **Security First**: Always validate inputs, sanitize outputs
- **Error Handling**: Comprehensive try-catch with proper logging
- **Performance**: Consider async/await for I/O operations

## Testing Requirements

### Minimum Coverage Requirements

- **Unit tests**: 80% code coverage minimum
- **Integration tests**: All API endpoints
- **E2E tests**: Critical user flows (chat, triage)

### Running Tests

```bash
# Backend tests
cd backend
pytest tests/ --cov=. --cov-report=html

# Frontend tests
cd frontend
npm test -- --coverage

# E2E tests
npm run test:e2e
```

### Writing Tests

- **Arrange-Act-Assert** pattern
- **Descriptive test names**: `test_safety_filter_blocks_prescriptive_language`
- **Mock external dependencies**: EHR, model inference
- **Test edge cases**: Empty inputs, invalid data, timeout scenarios

Example:
```python
def test_safety_filter_blocks_diagnosis():
    """Test that safety filter blocks diagnostic statements."""
    # Arrange
    filter = SafetyFilter(strictness="high")
    response = "You have pneumonia. Take antibiotics."
    
    # Act
    result = filter.check(response)
    
    # Assert
    assert result.is_safe == False
    assert "diagnostic language" in result.reason
    assert result.severity == "high"
```

## Documentation

### Required Documentation

1. **Code comments**: Complex logic, algorithms, workarounds
2. **Docstrings**: All public functions and classes
3. **API documentation**: OpenAPI/Swagger annotations
4. **README updates**: For new features or changes
5. **Architecture docs**: For structural changes

### Documentation Style

- Clear and concise
- Include examples where helpful
- Link to relevant resources
- Update diagrams if architecture changes

## Submitting Changes

### Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] No PHI or secrets in code
- [ ] Commit messages follow conventions
- [ ] Branch is up-to-date with main

### Pull Request Process

1. **Open an issue first** for major changes to discuss approach
2. **Create a PR** with clear title and description
3. **Fill out PR template** completely
4. **Link related issues** using keywords (Closes #123)
5. **Request review** from maintainers
6. **Address feedback** promptly and professionally

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe testing performed

## Checklist
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] No security issues
- [ ] Follows code style
```

## Review Process

### What Reviewers Look For

- **Correctness**: Does it work as intended?
- **Security**: Any PHI leaks or vulnerabilities?
- **Safety**: Medical AI safety considerations addressed?
- **Performance**: Any performance implications?
- **Maintainability**: Is code clean and well-documented?
- **Tests**: Adequate test coverage?

### Response Time

- Initial review: Within 48 hours (business days)
- Follow-up reviews: Within 24 hours
- Urgent fixes: Same day

### Approval Requirements

- Minimum 1 maintainer approval required
- All CI checks must pass
- No unresolved conversations
- Up-to-date with target branch

## Clinical & Safety Contributions

### Special Considerations

For contributions involving medical logic, safety filters, or clinical workflows:

1. **Clinical validation required**: Changes must be reviewed by clinical advisor
2. **Safety testing**: Extensive testing with edge cases
3. **Documentation**: Clear explanation of medical reasoning
4. **References**: Cite medical guidelines or research if applicable

### Safety-Critical Areas

Extra scrutiny for changes to:
- Safety filter logic
- Red flag detection
- PAR generation
- EHR data handling
- Consent workflows

## Community

### Getting Help

- **GitHub Discussions**: For questions and general discussion
- **GitHub Issues**: For bugs and feature requests
- **Email**: clinic-ai-team@yourclinic.example (placeholder)

### Suggesting Features

1. Search existing issues first
2. Open a detailed feature request
3. Include use case and rationale
4. Be open to feedback and alternatives

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes for significant contributions
- Project documentation

## Legal

By contributing, you agree that your contributions will be licensed under the MIT License and you have the right to submit the work under this license.

---

Thank you for contributing to Medical AI Assistant! Your efforts help make healthcare AI safer and more accessible.
