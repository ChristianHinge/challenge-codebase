FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Python dependencies
# -----------------------------
COPY requirements.txt .

# remove torchvision (not needed + removes warning)
RUN pip uninstall -y torchvision || true

RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy code
# -----------------------------
COPY . .

# -----------------------------
# Run inference
# -----------------------------
ENTRYPOINT ["python", "inference.py"]