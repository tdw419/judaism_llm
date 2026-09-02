# Judaism LLM RAG - Deployment Guide

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
