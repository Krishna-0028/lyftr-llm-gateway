DOCKER := $(shell command -v docker 2>/dev/null || echo /Applications/Docker.app/Contents/Resources/bin/docker)

.PHONY: up down logs test

up:
	$(DOCKER) compose up --build -d

down:
	$(DOCKER) compose down

logs:
	$(DOCKER) compose logs -f

test:
	cd backend && .venv/bin/pytest -v
