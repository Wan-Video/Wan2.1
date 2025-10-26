.PHONY: format docker-build docker-up docker-down docker-shell docker-logs docker-clean help

# Code formatting
format:
	isort generate.py gradio wan
	yapf -i -r *.py generate.py gradio wan

# Docker commands
docker-build:
	@echo "Building Docker image..."
	docker compose build

docker-up:
	@echo "Starting Wan2.1 container..."
	@mkdir -p models outputs cache
	docker compose up -d wan2-1

docker-down:
	@echo "Stopping Wan2.1 container..."
	docker compose down

docker-restart: docker-down docker-up

docker-shell:
	@echo "Entering container shell..."
	docker compose exec wan2-1 bash

docker-logs:
	@echo "Showing container logs..."
	docker compose logs -f wan2-1

docker-clean:
	@echo "Cleaning up Docker resources..."
	docker compose down -v
	docker system prune -f

docker-status:
	@echo "Container status:"
	docker compose ps

# Help command
help:
	@echo "Wan2.1 Makefile Commands:"
	@echo ""
	@echo "Code Formatting:"
	@echo "  make format          - Format Python code with isort and yapf"
	@echo ""
	@echo "Docker Management:"
	@echo "  make docker-build    - Build Docker image"
	@echo "  make docker-up       - Start container (with GPU support)"
	@echo "  make docker-down     - Stop container"
	@echo "  make docker-restart  - Restart container"
	@echo "  make docker-shell    - Enter container shell"
	@echo "  make docker-logs     - View container logs"
	@echo "  make docker-status   - Show container status"
	@echo "  make docker-clean    - Remove containers and clean up"
	@echo ""
	@echo "For detailed Docker setup, see DOCKER_SETUP.md"
