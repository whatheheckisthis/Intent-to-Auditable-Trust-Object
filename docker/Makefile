.PHONY: all up down clean certs

# Compose files
COMPOSE_FILE=compose/docker-compose.yml

# Default target: build and bring up the stack
all: certs up

# Start stack
up:
	docker-compose -f $(COMPOSE_FILE) up -d

# Stop stack
down:
	docker-compose -f $(COMPOSE_FILE) down

# Clean volumes
clean:
	docker-compose -f $(COMPOSE_FILE) down -v
	rm -rf grafana/provisioning/dashboards/*.json

# Generate self-signed TLS certificates for Nginx
certs:
	@mkdir -p nginx/ssl
	openssl req -x509 -nodes -days 365 \
		-newkey rsa:2048 \
		-keyout nginx/ssl/iato.key \
		-out nginx/ssl/iato.crt \
		-subj "/C=NZ/ST=Auckland/L=Auckland/O=IATO/CN=localhost"
	@echo "TLS certs generated at nginx/ssl/"
