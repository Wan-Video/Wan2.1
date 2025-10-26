#!/bin/bash
# Wan2.1 Docker Helper Script
# Quick start script for running Wan2.1 in Docker

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CONTAINER_NAME="wan2.1-gpu"
IMAGE_NAME="wan2.1:latest"
MODELS_DIR="./models"
OUTPUTS_DIR="./outputs"
CACHE_DIR="./cache"

# Print header
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Wan2.1 Docker Helper Script${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Please install Docker first: https://docs.docker.com/engine/install/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# Function to check NVIDIA Docker runtime
check_nvidia_runtime() {
    if ! docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}Warning: NVIDIA Docker runtime not available${NC}"
        echo "GPU acceleration will not be available"
        echo "Install NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        return 1
    fi
    return 0
}

# Function to create directories
create_directories() {
    echo -e "${BLUE}Creating required directories...${NC}"
    mkdir -p "$MODELS_DIR" "$OUTPUTS_DIR" "$CACHE_DIR"
    echo -e "${GREEN}✓ Directories created${NC}\n"
}

# Function to build Docker image
build_image() {
    echo -e "${BLUE}Building Docker image...${NC}"
    docker compose build
    echo -e "${GREEN}✓ Docker image built successfully${NC}\n"
}

# Function to start container
start_container() {
    echo -e "${BLUE}Starting Wan2.1 container...${NC}"

    if check_nvidia_runtime; then
        echo "Starting with GPU support..."
        docker compose up -d wan2-1
    else
        echo "Starting in CPU-only mode..."
        docker compose --profile cpu up -d wan2-1-cpu
        CONTAINER_NAME="wan2.1-cpu"
    fi

    echo -e "${GREEN}✓ Container started successfully${NC}\n"
}

# Function to show status
show_status() {
    echo -e "${BLUE}Container Status:${NC}"
    docker compose ps
    echo ""
}

# Function to show logs
show_logs() {
    echo -e "${BLUE}Showing container logs (Ctrl+C to exit)...${NC}"
    docker compose logs -f "$CONTAINER_NAME"
}

# Function to enter container
enter_container() {
    echo -e "${BLUE}Entering container shell...${NC}"
    docker compose exec wan2-1 bash || docker compose exec wan2-1-cpu bash
}

# Function to stop container
stop_container() {
    echo -e "${BLUE}Stopping container...${NC}"
    docker compose down
    echo -e "${GREEN}✓ Container stopped${NC}\n"
}

# Function to show help
show_help() {
    cat << EOF
Usage: ./docker-run.sh [COMMAND]

Commands:
    build       Build the Docker image
    start       Start the container
    stop        Stop the container
    restart     Restart the container
    status      Show container status
    logs        Show container logs
    shell       Enter container shell
    clean       Stop container and clean up
    help        Show this help message

Examples:
    ./docker-run.sh build        # Build the image
    ./docker-run.sh start        # Start the container
    ./docker-run.sh shell        # Enter the container
    ./docker-run.sh logs         # View logs

For detailed documentation, see DOCKER_SETUP.md
EOF
}

# Main script logic
case "${1:-start}" in
    build)
        create_directories
        build_image
        ;;
    start)
        create_directories
        if ! docker images | grep -q "$IMAGE_NAME"; then
            build_image
        fi
        start_container
        show_status
        echo -e "${GREEN}Container is running!${NC}"
        echo -e "Run ${BLUE}./docker-run.sh shell${NC} to enter the container"
        echo -e "Run ${BLUE}./docker-run.sh logs${NC} to view logs"
        ;;
    stop)
        stop_container
        ;;
    restart)
        stop_container
        start_container
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    shell)
        enter_container
        ;;
    clean)
        echo -e "${YELLOW}This will stop the container and remove volumes${NC}"
        read -p "Are you sure? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker compose down -v
            echo -e "${GREEN}✓ Cleanup complete${NC}"
        fi
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac
