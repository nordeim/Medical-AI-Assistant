# Innovation Prototype

This is an AI-powered innovation prototype built with the Rapid Prototyping Engine.

## Features

- FastAPI backend
- Modern frontend
- Docker containerization
- Automated testing
- CI/CD pipeline

## Quick Start

1. Build and run with Docker:
   ```bash
   docker-compose up --build
   ```

2. Access the application:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   cd frontend && npm install
   ```

2. Run development servers:
   ```bash
   # Backend
   uvicorn main:app --reload
   
   # Frontend
   cd frontend && npm start
   ```

## Testing

Run tests:
```bash
pytest tests/
npm test
```

## Deployment

The prototype supports deployment to:
- Development environment
- Testing environment  
- Production environment

See deployment documentation for details.
