# Start from a CUDA-capable Python image for GPU deployments.
FROM nvidia/cuda:12.3.0-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    git build-essential \
    libjpeg-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Use python3 as default python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Create app directory
WORKDIR /app

# Copy requirements and all project files
COPY requirements.txt ./
COPY .env .env
COPY config.py ./
COPY launch_medical_mcp.py ./
COPY medical_mcp_client.py ./
COPY medical_mcp_server.py ./
COPY SFT ./SFT
COPY UI_src_codes ./UI_src_codes

# (Optional: include other required files or folders)

# Install Python dependencies
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Create (and own) uploads dir
RUN mkdir -p /app/UI_src_codes/uploads

# Expose FastAPI port
EXPOSE 8000

# Set environment variable so torch uses the right device (OPTIONAL)
ENV CUDA_VISIBLE_DEVICES=0

# Start the UI web server (production: no --reload)
CMD ["python", "UI_src_codes/launch_ui.py"]
