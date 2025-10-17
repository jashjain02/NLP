# Docker Deployment Guide

This guide explains how to build, test, and deploy the NLP Finance Pipeline using Docker.

## 🐳 Quick Start

### Build and Run Locally

```bash
# Build the Docker image
docker build -t nlp-finance:latest .

# Run the container
docker run -p 8501:8501 -p 8000:8000 nlp-finance:latest
```

### Using Docker Compose

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 📋 Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Git

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# API Keys
NEWSAPI_KEY=your_newsapi_key_here
MARKETAUX_API_KEY=your_marketaux_key_here

# Optional: Database configuration
# DATABASE_URL=postgresql://user:password@localhost:5432/nlp_finance
```

### Docker Compose Configuration

The `docker-compose.yml` file includes:

- **Ports**: 8501 (Streamlit), 8000 (FastAPI)
- **Volumes**: Persistent storage for data, models, and reports
- **Health Checks**: Automatic health monitoring
- **Restart Policy**: Automatic restart on failure

## 🚀 Deployment Options

### 1. Local Development

```bash
# Clone repository
git clone https://github.com/jashjain02/NLP.git
cd NLP

# Build and run
docker-compose up -d

# Access the application
open http://localhost:8501
```

### 2. Production Deployment

```bash
# Use production Dockerfile
docker build -f Dockerfile.prod -t nlp-finance:prod .

# Run with production settings
docker run -d \
  --name nlp-finance-prod \
  -p 8501:8501 \
  -p 8000:8000 \
  -e NEWSAPI_KEY=your_key \
  -e MARKETAUX_API_KEY=your_key \
  -v nlp-finance-data:/app/data \
  -v nlp-finance-models:/app/models \
  -v nlp-finance-reports:/app/reports \
  nlp-finance:prod
```

### 3. Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/deployment.yaml

# Create secrets
kubectl create secret generic nlp-finance-secrets \
  --from-literal=newsapi-key=your_key \
  --from-literal=marketaux-key=your_key

# Check deployment status
kubectl get pods -l app=nlp-finance-app
```

## 🧪 Testing

### Automated Testing

```bash
# Run the build script with tests
./scripts/docker-build.sh

# Test with specific tag
./scripts/docker-build.sh --tag v1.0.0

# Test production build
./scripts/docker-build.sh --prod
```

### Manual Testing

```bash
# Start container
docker run -d --name nlp-test -p 8501:8501 nlp-finance:latest

# Wait for startup
sleep 30

# Test health endpoint
curl http://localhost:8501/_stcore/health

# Test Streamlit interface
curl http://localhost:8501

# Check logs
docker logs nlp-test

# Clean up
docker stop nlp-test && docker rm nlp-test
```

## 📊 Monitoring

### Health Checks

The application includes built-in health checks:

- **Streamlit**: `http://localhost:8501/_stcore/health`
- **FastAPI**: `http://localhost:8000/health` (if implemented)

### Logs

```bash
# View container logs
docker logs nlp-finance-app

# Follow logs in real-time
docker logs -f nlp-finance-app

# View logs with timestamps
docker logs -t nlp-finance-app
```

### Resource Usage

```bash
# Check container stats
docker stats nlp-finance-app

# Check resource usage
docker exec nlp-finance-app top
```

## 🔒 Security

### Security Best Practices

1. **Non-root User**: Production images run as non-root user
2. **Minimal Base Image**: Uses slim Python image
3. **Security Scanning**: Automated vulnerability scanning
4. **Secrets Management**: Environment variables for sensitive data

### Security Scanning

```bash
# Scan for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image nlp-finance:latest

# Scan with specific format
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image --format table nlp-finance:latest
```

## 🚀 CI/CD Integration

### GitHub Actions

The repository includes automated Docker builds:

- **Build**: On every push to main/develop
- **Test**: Automated testing of Docker images
- **Security**: Vulnerability scanning
- **Deploy**: Automatic deployment to staging

### Build Triggers

- **Push to main**: Build and push to registry
- **Tag push**: Create release images
- **Pull Request**: Build and test only

## 📈 Performance

### Resource Requirements

**Minimum:**
- CPU: 250m
- Memory: 512Mi

**Recommended:**
- CPU: 1000m
- Memory: 2Gi

### Optimization

1. **Multi-stage Build**: Reduces image size
2. **Layer Caching**: Faster subsequent builds
3. **Health Checks**: Automatic recovery
4. **Resource Limits**: Prevents resource exhaustion

## 🐛 Troubleshooting

### Common Issues

1. **Port Conflicts**: Ensure ports 8501 and 8000 are available
2. **Memory Issues**: Increase Docker memory limit
3. **Permission Errors**: Check file permissions in volumes
4. **API Key Issues**: Verify environment variables

### Debug Commands

```bash
# Check container status
docker ps -a

# Inspect container
docker inspect nlp-finance-app

# Execute shell in container
docker exec -it nlp-finance-app /bin/bash

# Check logs
docker logs nlp-finance-app --tail 100
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🤝 Support

For issues and questions:

1. Check the [GitHub Issues](https://github.com/jashjain02/NLP/issues)
2. Review the logs: `docker logs nlp-finance-app`
3. Test locally: `./scripts/docker-build.sh`
4. Create a new issue with logs and configuration details
