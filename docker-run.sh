#!/bin/bash
# Script to build and run the GraphRAG API Service using Docker

# Exit on error
set -e

# Make script executable
chmod +x docker-run.sh

# Function to show help
show_help() {
  echo "Usage: ./docker-run.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --build          Build or rebuild the Docker image"
  echo "  --start          Start the service"
  echo "  --stop           Stop the service"
  echo "  --restart        Restart the service"
  echo "  --logs           Show logs"
  echo "  --status         Show status"
  echo "  --help           Show this help"
  echo ""
  echo "Examples:"
  echo "  ./docker-run.sh --build --start   Build and start the service"
  echo "  ./docker-run.sh --logs            Show logs"
}

# Process arguments
BUILD=0
START=0
STOP=0
RESTART=0
LOGS=0
STATUS=0

# If no arguments, show help
if [ $# -eq 0 ]; then
  show_help
  exit 0
fi

# Parse arguments
while [ "$1" != "" ]; do
  case $1 in
    --build)
      BUILD=1
      ;;
    --start)
      START=1
      ;;
    --stop)
      STOP=1
      ;;
    --restart)
      RESTART=1
      ;;
    --logs)
      LOGS=1
      ;;
    --status)
      STATUS=1
      ;;
    --help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
  shift
done

# Build
if [ $BUILD -eq 1 ]; then
  echo "Building GraphRAG API Service Docker image..."
  docker-compose build
fi

# Stop
if [ $STOP -eq 1 ]; then
  echo "Stopping GraphRAG API Service..."
  docker-compose down
fi

# Start
if [ $START -eq 1 ]; then
  echo "Starting GraphRAG API Service..."
  docker-compose up -d
  echo "GraphRAG API Service is running on http://localhost:6989"
  echo "API documentation available at http://localhost:6989/v1/docs"
fi

# Restart
if [ $RESTART -eq 1 ]; then
  echo "Restarting GraphRAG API Service..."
  docker-compose restart
  echo "GraphRAG API Service restarted"
fi

# Logs
if [ $LOGS -eq 1 ]; then
  echo "Showing logs for GraphRAG API Service..."
  docker-compose logs -f
fi

# Status
if [ $STATUS -eq 1 ]; then
  echo "Status of GraphRAG API Service:"
  docker-compose ps
fi
