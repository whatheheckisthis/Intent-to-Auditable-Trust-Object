# Use an official lightweight Python image as base
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install core dependencies
RUN apt-get update && apt-get install -y \
    git \
    openssl \
    bash \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the entire repo source code into the container
COPY . .

# Make the shell scripts executable
RUN chmod +x *.sh scripts/*.sh ci/*.sh

# Create and activate Python virtual environment within container
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip inside virtual environment
RUN pip install --upgrade pip

# (Optional) Install Python dependencies if environment.yml or requirements.txt present
# RUN pip install -r requirements.txt

# Run initialization script to check environment and generate changelog
RUN ./init_environment.sh

# Default command to launch when container starts (adjust as necessary)
CMD ["/bin/bash"]
