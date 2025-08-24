FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openssl \
        ca-certificates \
        curl \
        bash && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements if defined in config
COPY ../config/requirements.txt /app/config/requirements.txt
RUN pip install --no-cache-dir -r /app/config/requirements.txt || true

CMD ["bash"]