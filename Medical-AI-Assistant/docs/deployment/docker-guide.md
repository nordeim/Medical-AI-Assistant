# Docker Deployment Guide

## Quick Start

### Prerequisites

- Docker 24.0+ installed
- Docker Compose 2.20+ installed
- 8GB RAM minimum (16GB recommended)
- 20GB free disk space

### Development Environment

1. **Clone and setup**
   ```bash
   git clone https://github.com/nordeim/Medical-AI-Assistant.git
   cd Medical-AI-Assistant
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Start all services**
   ```bash
   docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up --build
   ```

3. **Access services**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - pgAdmin: http://localhost:5050
   - Mailhog: http://localhost:8025

### Production Environment

1. **Build images**
   ```bash
   docker compose -f docker/docker-compose.yml build
   ```

2. **Start services**
   ```bash
   docker compose -f docker/docker-compose.yml up -d
   ```

3. **Enable nginx**
   ```bash
   docker compose --profile production -f docker/docker-compose.yml up -d
   ```

## Service Architecture

- **db**: PostgreSQL 17 Alpine - Primary database
- **redis**: Redis 7 Alpine - Caching and rate limiting
- **backend**: Python 3.11 - FastAPI + LangChain
- **frontend**: Node 22 Alpine - React application
- **nginx**: Nginx Alpine - Reverse proxy (production)

## Common Commands

### Logs
```bash
# View all logs
docker compose -f docker/docker-compose.yml logs -f

# View specific service
docker compose -f docker/docker-compose.yml logs -f backend

# Last 100 lines
docker compose -f docker/docker-compose.yml logs --tail=100
```

### Database

```bash
# Access PostgreSQL
docker compose -f docker/docker-compose.yml exec db psql -U meduser -d meddb

# Run migrations
docker compose -f docker/docker-compose.yml exec backend alembic upgrade head

# Create migration
docker compose -f docker/docker-compose.yml exec backend alembic revision --autogenerate -m "description"

# Database backup
docker compose -f docker/docker-compose.yml exec db pg_dump -U meduser meddb > backup.sql

# Database restore
cat backup.sql | docker compose -f docker/docker-compose.yml exec -T db psql -U meduser -d meddb
```

### Cleanup

```bash
# Stop all services
docker compose -f docker/docker-compose.yml down

# Stop and remove volumes
docker compose -f docker/docker-compose.yml down -v

# Remove images
docker compose -f docker/docker-compose.yml down --rmi all
```

## Health Checks

All services include health checks:
- Database: `pg_isready` check every 10s
- Backend: HTTP health endpoint every 30s
- Frontend: HTTP check every 30s
- Redis: `ping` command every 10s

Check service health:
```bash
docker compose -f docker/docker-compose.yml ps
```

## Troubleshooting

### Backend won't start
```bash
# Check logs
docker compose -f docker/docker-compose.yml logs backend

# Verify database connection
docker compose -f docker/docker-compose.yml exec db pg_isready -U meduser

# Restart backend
docker compose -f docker/docker-compose.yml restart backend
```

### Frontend build fails
```bash
# Clear node_modules
docker compose -f docker/docker-compose.yml exec frontend rm -rf node_modules
docker compose -f docker/docker-compose.yml restart frontend
```

### Port conflicts
```bash
# Find process using port
lsof -i :8000  # or :3000, :5432

# Stop conflicting services or change ports in .env
```

## Security Considerations

1. **Change default passwords** in `.env`
2. **Use secrets** for production credentials
3. **Enable TLS** for nginx in production
4. **Restrict network access** using firewall rules
5. **Regular updates** of base images

## Performance Tuning

### Database
- Adjust `shared_buffers` in PostgreSQL config
- Enable connection pooling in backend
- Use read replicas for scaling

### Backend
- Increase worker count: `API_WORKERS=8`
- Enable response caching
- Use load balancer for multiple instances

### Frontend
- Enable CDN for static assets
- Use nginx caching
- Optimize bundle size

## Monitoring

Add monitoring services (optional):

```yaml
# docker-compose.monitoring.yml
services:
  prometheus:
    image: prom/prometheus
    # ... configuration
  
  grafana:
    image: grafana/grafana
    # ... configuration
```

Start with monitoring:
```bash
docker compose \
  -f docker/docker-compose.yml \
  -f docker/docker-compose.monitoring.yml \
  up -d
```
