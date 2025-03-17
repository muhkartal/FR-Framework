# Docker Guide for FR-Framework

This guide covers how to build, run, and manage Docker containers for the FR-Framework.

## Prerequisites

-  [Docker](https://www.docker.com/get-started)
-  [Docker Compose](https://docs.docker.com/compose/install/)

## Components

The FR-Framework Docker setup consists of two main services:

1. **fr-api**: The FastAPI-based REST API server
2. **fr-web**: A web interface for using the FR-Framework (optional)

## Quick Start

### Starting the Services

```bash
# Build and start all services
docker-compose up -d

# Build and start only the API
docker-compose up -d fr-api
```

### Stopping the Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Configuration

### Environment Variables

You can configure the services using environment variables in the `docker-compose.yml` file:

**fr-api:**

-  `MODEL_PATH`: Path to the model files inside the container
-  `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
-  `MAX_WORKERS`: Number of worker processes for the API server

**fr-web:**

-  `API_URL`: URL of the API service

### Volumes

The following volumes are mounted for persistent storage:

-  `./api/output:/app/api/output`: Directory for storing output files
-  `./api/temp:/app/api/temp`: Directory for temporary files
-  `./models:/app/models`: Directory for model files

## Building Images

### Building Individual Images

```bash
# Build the API image
docker build -t fr-framework/api .

# Build the web interface image
docker build -t fr-framework/web -f web/Dockerfile web/
```

### Customizing Builds

You can customize the Docker builds by modifying the Dockerfiles:

-  `Dockerfile`: Main API service
-  `web/Dockerfile`: Web interface

## Deployment

### Production Deployment

For production deployment, consider the following adjustments:

1. Use specific image tags rather than relying on auto-builds
2. Set up proper SSL termination with a reverse proxy like Nginx or Traefik
3. Configure proper authentication for the API
4. Use Docker secrets for sensitive configuration

Example production docker-compose.yml adjustments:

```yaml
version: "3.8"

services:
   fr-api:
      image: fr-framework/api:v1.0.0
      restart: always
      deploy:
         resources:
            limits:
               cpus: "2"
               memory: 4G
      environment:
         - LOG_LEVEL=WARNING
         - MAX_WORKERS=8

   fr-web:
      image: fr-framework/web:v1.0.0
      restart: always
```

### Health Checks

The API service includes a health check that verifies the API is responding correctly:

```yaml
healthcheck:
   test: ["CMD", "curl", "-f", "http://localhost:8000/"]
   interval: 30s
   timeout: 10s
   retries: 3
   start_period: 40s
```

## Troubleshooting

### Common Issues

1. **Container fails to start**:

   -  Check logs: `docker-compose logs fr-api`
   -  Ensure all required volumes are mounted correctly
   -  Verify model files are present

2. **Performance issues**:

   -  Increase resource limits in docker-compose.yml
   -  Consider using GPU acceleration (see below)

3. **API not accessible**:
   -  Check network configuration
   -  Verify port mappings in docker-compose.yml

### Using GPU Acceleration

To use GPU acceleration for the CNN face detection model:

1. Install [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
2. Modify docker-compose.yml:

```yaml
services:
   fr-api:
      # ... other settings
      deploy:
         resources:
            reservations:
               devices:
                  - driver: nvidia
                    count: 1
                    capabilities: [gpu]
```

## Maintenance

### Updating Images

To update to a new version:

```bash
# Pull latest code
git pull

# Rebuild and restart containers
docker-compose up -d --build
```

### Backing Up Data

To backup generated data:

```bash
# Backup output directory
tar -czf fr-framework-output-backup.tar.gz api/output/

# Backup models
tar -czf fr-framework-models-backup.tar.gz models/
```
