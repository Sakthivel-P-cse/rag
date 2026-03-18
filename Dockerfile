# Container image for deploying the RAG service on GCP Cloud Run

FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Create and switch to the working directory
WORKDIR /app

# Install system deps (if FAISS or other libs need BLAS, etc.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repository
COPY . .

# Cloud Run expects the service to listen on $PORT (default 8080)
ENV PORT=8080

# Expose the port for local testing (not required by Cloud Run)
EXPOSE 8080

# Start the FastAPI app via Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
