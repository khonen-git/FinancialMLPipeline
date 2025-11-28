FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY templates/ ./templates/
COPY run_experiment.py .
COPY setup.py .

# Install package
RUN pip install -e .

# Create directories
RUN mkdir -p data logs outputs

# Set environment
ENV PYTHONPATH=/app
ENV HYDRA_FULL_ERROR=1

# Entry point
CMD ["python", "run_experiment.py"]

