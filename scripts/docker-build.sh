#!/bin/bash

# Docker build script for NLP Finance Pipeline
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="nlp-finance"
TAG="latest"
DOCKERFILE="Dockerfile"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --tag)
      TAG="$2"
      shift 2
      ;;
    --dockerfile)
      DOCKERFILE="$2"
      shift 2
      ;;
    --prod)
      DOCKERFILE="Dockerfile.prod"
      shift
      ;;
    --help)
      echo "Usage: $0 [--tag TAG] [--dockerfile DOCKERFILE] [--prod] [--help]"
      echo "  --tag TAG        Docker image tag (default: latest)"
      echo "  --dockerfile     Dockerfile to use (default: Dockerfile)"
      echo "  --prod           Use production Dockerfile"
      echo "  --help           Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

echo -e "${BLUE}🐳 Building Docker image: ${IMAGE_NAME}:${TAG}${NC}"
echo -e "${YELLOW}Using Dockerfile: ${DOCKERFILE}${NC}"

# Build the Docker image
echo -e "${BLUE}📦 Building image...${NC}"
docker build -f "$DOCKERFILE" -t "${IMAGE_NAME}:${TAG}" .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Docker image built successfully!${NC}"
else
    echo -e "${RED}❌ Docker build failed!${NC}"
    exit 1
fi

# Test the image
echo -e "${BLUE}🧪 Testing Docker image...${NC}"

# Run the container in background
echo -e "${YELLOW}Starting container...${NC}"
docker run -d --name "${IMAGE_NAME}-test" -p 8501:8501 -p 8000:8000 "${IMAGE_NAME}:${TAG}"

# Wait for the app to start
echo -e "${YELLOW}Waiting for app to start...${NC}"
sleep 30

# Test health check
echo -e "${BLUE}Testing health check...${NC}"
if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Health check passed!${NC}"
else
    echo -e "${RED}❌ Health check failed!${NC}"
    echo -e "${YELLOW}Container logs:${NC}"
    docker logs "${IMAGE_NAME}-test"
    docker stop "${IMAGE_NAME}-test" > /dev/null 2>&1
    docker rm "${IMAGE_NAME}-test" > /dev/null 2>&1
    exit 1
fi

# Test Streamlit interface
echo -e "${BLUE}Testing Streamlit interface...${NC}"
if curl -f http://localhost:8501 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Streamlit interface accessible!${NC}"
else
    echo -e "${RED}❌ Streamlit interface not accessible!${NC}"
fi

# Clean up
echo -e "${YELLOW}Cleaning up...${NC}"
docker stop "${IMAGE_NAME}-test" > /dev/null 2>&1
docker rm "${IMAGE_NAME}-test" > /dev/null 2>&1

echo -e "${GREEN}🎉 Docker image test completed successfully!${NC}"
echo -e "${BLUE}Image: ${IMAGE_NAME}:${TAG}${NC}"
echo -e "${YELLOW}To run the container:${NC}"
echo -e "  docker run -p 8501:8501 -p 8000:8000 ${IMAGE_NAME}:${TAG}"
echo -e "${YELLOW}To run with docker-compose:${NC}"
echo -e "  docker-compose up"
