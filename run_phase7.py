#!/usr/bin/env python3
"""
Phase 7: Deployment
Package for deployment with Docker and environment variables
"""

import os
import sys
from pathlib import Path

def create_dockerfile():
    """Create Dockerfile for RAG system."""
    dockerfile = """
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY run_phase5.py .
COPY judaism-llm-qwen2.5-7b-merged/ ./model/
COPY chroma_db/ ./chroma_db/

# Set environment variables
ENV MODEL_PATH=./model
ENV CHROMA_DIR=./chroma_db
ENV PORT=8000

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "run_phase5:app", "--host", "0.0.0.0", "--port", "8000"]
"""

    with open("Dockerfile", "w") as f:
        f.write(dockerfile)

    print("✓ Created Dockerfile")

def create_requirements():
    """Create requirements.txt for deployment."""
    requirements = """
# RAG System Requirements
chromadb>=1.5.0
sentence-transformers>=2.2.0
transformers>=4.30.0
torch>=2.0.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
numpy>=1.24.0
accelerate>=0.20.0
bitsandbytes>=0.41.0
"""

    with open("requirements.txt", "w") as f:
        f.write(requirements)

    print("✓ Created requirements.txt")

def create_docker_compose():
    """Create docker-compose.yml for multi-container deployment."""
    compose = """
version: '3.8'

services:
  judaism-rag:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=./model
      - CHROMA_DIR=./chroma_db
      - TOP_K=5
    volumes:
      - ./chroma_db:/app/chroma_db
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Optional: Nginx reverse proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - judaism-rag
    restart: unless-stopped
"""

    with open("docker-compose.yml", "w") as f:
        f.write(compose)

    print("✓ Created docker-compose.yml")

def create_deployment_guide():
    """Create deployment guide."""
    guide = """# Judaism LLM RAG - Deployment Guide

## Quick Start (Docker)

### Build and Run

```bash
# Build Docker image
docker build -t judaism-llm-rag .

# Run container
docker run -p 8000:8000 judaism-llm-rag

# Access at http://localhost:8000
```

### Docker Compose (Production)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f judaism-rag

# Stop services
docker-compose down
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./model` | Path to Judaism LLM model |
| `CHROMA_DIR` | `./chroma_db` | Path to ChromaDB data |
| `PORT` | `8000` | Web server port |
| `TOP_K` | `5` | Number of retrieved passages |

## Health Check

```bash
curl http://localhost:8000/
```

Expected: HTML interface

## Monitoring

### Logs

```bash
# Docker logs
docker logs -f judaism-llm-rag

# Docker Compose logs
docker-compose logs -f judaism-rag
```

### Metrics

- Response time: < 5 seconds
- Success rate: > 80%
- Source accuracy: > 70%

## Production Checklist

- [ ] Set appropriate CPU/RAM limits
- [ ] Configure persistent storage for ChromaDB
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure log rotation
- [ ] Set up HTTPS (Nginx reverse proxy)
- [ ] Configure rate limiting
- [ ] Set up backup for ChromaDB

## Backup and Recovery

### Backup ChromaDB

```bash
# Backup
docker exec judaism-llm-rag tar -czf /tmp/chroma_backup.tar.gz /app/chroma_db
docker cp judaism-llm-rag:/tmp/chroma_backup.tar.gz ./backups/

# Restore
docker cp ./backups/chroma_backup.tar.gz judaism-llm-rag:/tmp/
docker exec judaism-llm-rag tar -xzf /tmp/chroma_backup.tar.gz -C /app/
```

## Troubleshooting

### Container won't start

```bash
# Check logs
docker logs judaism-llm-rag

# Check resources
docker stats judaism-llm-rag
```

### Out of memory

- Increase Docker memory limit
- Reduce `TOP_K` environment variable
- Use smaller model (Phase 1 with 3B instead of 7B)

### Slow responses

- Check GPU availability: `nvidia-smi`
- Reduce `TOP_K` from 5 to 3
- Increase batch size in ChromaDB

---

**Created:** September 1, 2026
**Version:** 1.0.0
"""

    with open("DEPLOYMENT.md", "w") as f:
        f.write(guide)

    print("✓ Created DEPLOYMENT.md")

def main():
    """Run Phase 7: Deployment setup."""
    print("Phase 7: Deployment\n")

    # Create deployment files
    create_dockerfile()
    create_requirements()
    create_docker_compose()
    create_deployment_guide()

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    print("\n✓ Phase 7 Complete:")
    print("  Dockerfile created")
    print("  requirements.txt created")
    print("  docker-compose.yml created")
    print("  DEPLOYMENT.md created")
    print("\nNext steps:")
    print("  docker build -t judaism-llm-rag .")
    print("  docker-compose up -d")
    print("  curl http://localhost:8000/")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)