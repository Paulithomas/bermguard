# BermGuard AI
# Plataforma fijada a amd64: el desarrollo es en Apple Silicon (arm64) pero
# el despliegue objetivo es x86_64 con GPU NVIDIA.
FROM --platform=linux/amd64 python:3.11-slim

# Sin descargas en runtime (SUP-24): las librerias no deben resolver remoto.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/app/weights/torch \
    HF_HOME=/app/weights/hf \
    YOLO_CONFIG_DIR=/app/weights/yolo \
    YOLO_AUTOINSTALL=false \
    MPLCONFIGDIR=/tmp/matplotlib

# Dependencias de sistema para OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Capa de dependencias separada: se cachea entre builds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/weights /app/test /app/output

COPY main.py .
COPY configs/ ./configs/
COPY src/ ./src/

COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["--input", "/app/test", "--output", "/app/output", "--method", "1"]