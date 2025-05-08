#!/bin/bash
# Launch script for the GraphRAG API Service

# Exit on error
set -e

# Make script executable
chmod +x launch.sh

# Function to show help
show_help() {
  echo "GraphRAG API Service Launcher"
  echo "Usage: ./launch.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --local         Run the service locally (default)"
  echo "  --docker        Run the service using Docker"
  echo "  --setup         Run the setup script first"
  echo "  --port PORT     Specify a custom port (default: 6989)"
  echo "  --help          Show this help"
  echo ""
  echo "Examples:"
  echo "  ./launch.sh --local --setup           Setup and run locally"
  echo "  ./launch.sh --docker --port 8080      Run with Docker on port 8080"
}

# Set defaults
RUN_MODE="local"
RUN_SETUP=0
PORT=6989

# Parse arguments
while [ "$1" != "" ]; do
  case $1 in
    --local)
      RUN_MODE="local"
      ;;
    --docker)
      RUN_MODE="docker"
      ;;
    --setup)
      RUN_SETUP=1
      ;;
    --port)
      shift
      PORT=$1
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

# Display banner
echo "==============================================="
echo "  GraphRAG API Service"
echo "  Mode: $RUN_MODE | Port: $PORT"
echo "==============================================="

# Run in local mode
if [ "$RUN_MODE" = "local" ]; then
  # Run setup if requested
  if [ $RUN_SETUP -eq 1 ]; then
    echo "Running setup script..."
    chmod +x setup.sh
    ./setup.sh
  fi
  
  # Create static directory if it doesn't exist
  mkdir -p static
  
  # Modify port in run script if needed
  if [ $PORT -ne 6989 ]; then
    echo "Using custom port: $PORT"
    # Use temporary file to run with custom port
    cat > run_temp.sh << EOL
#!/bin/bash
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port $PORT --reload
EOL
    chmod +x run_temp.sh
    ./run_temp.sh
    rm run_temp.sh
  else
    # Run with default port
    chmod +x run.sh
    ./run.sh
  fi
  
# Run in Docker mode
elif [ "$RUN_MODE" = "docker" ]; then
  # Update docker-compose.yml with custom port if needed
  if [ $PORT -ne 6989 ]; then
    echo "Using custom port: $PORT"
    # Use sed to replace the port in docker-compose.yml (create backup first)
    cp docker-compose.yml docker-compose.yml.bak
    sed -i.bak "s/6989:6989/$PORT:6989/g" docker-compose.yml
  fi
  
  # Run with Docker
  chmod +x docker-run.sh
  
  if [ $RUN_SETUP -eq 1 ]; then
    ./docker-run.sh --build --start
  else
    ./docker-run.sh --start
  fi
  
  # Show service URL
  echo "GraphRAG API Service is now running at: http://localhost:$PORT"
  echo "API documentation available at: http://localhost:$PORT/v1/docs"
  
  # Restore original docker-compose.yml if it was modified
  if [ $PORT -ne 6989 ]; then
    mv docker-compose.yml.bak docker-compose.yml
  fi
else
  echo "Invalid run mode: $RUN_MODE"
  show_help
  exit 1
fi
