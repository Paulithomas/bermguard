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

# Dependencias de sistema: libglib y libgl para OpenCV headless, git para
# instalar el paquete clip de OpenAI, que no esta publicado en PyPI y es
# requerido por el encoder de texto de YOLO-World.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Capa de dependencias separada: se cachea entre builds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/test /app/output

# Descarga de pesos en build-time (SUP-24). Incluye el encoder CLIP que
# YOLO-World resuelve al fijar el vocabulario: sin este paso el metodo 2
# intentaria descargarlo en runtime y fallaria en un contenedor sin red.
COPY configs/ ./configs/
COPY tools/prepare_weights.py ./tools/
RUN python tools/prepare_weights.py

COPY main.py .
COPY src/ ./src/
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["--input", "/app/test", "--output", "/app/output", "--method", "1"]