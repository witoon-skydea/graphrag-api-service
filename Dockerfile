FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    tesseract-ocr \
    libtesseract-dev \
    tesseract-ocr-eng \
    libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p db companies graph prompts data/uploads static temp_mm_chat

# Expose the API port
EXPOSE 6989

# Command to run the API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "6989"]
