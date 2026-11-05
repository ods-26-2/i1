# Contrato P2 1.1 (SIS-1): Python >=3.10. Python 3.11 permite usar PyGObject/GStreamer do Debian.
FROM python:3.11-slim-bookworm

WORKDIR /app

# Dependências de sistema para OpenCV, protocolo P4R1 via GStreamer/PyGObject e librdkafka
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgstreamer1.0-0 \
    python3-gi \
    gir1.2-gstreamer-1.0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    && rm -rf /var/lib/apt/lists/*

# Instala torch CPU antes do ultralytics para evitar o download de ~3.5 GB de bibliotecas CUDA no container CPU
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Instala dependências para execução, validação, Kafka e a biblioteca oficial P2 (SIS-1)
RUN pip install --no-cache-dir \
    "numpy>=1.21.6,<2" \
    "opencv-python-headless<5" \
    pyyaml \
    jsonschema \
    confluent-kafka \
    ultralytics \
    onnx \
    onnxruntime \
    "p2[cpu] @ https://github.com/ods-26-2/p2/releases/download/v0.2.0a1/p2-0.2.0a1-py3-none-any.whl"

COPY I1/ /app/I1/
COPY schemas/ /app/schemas/

ENV PYTHONPATH=/usr/lib/python3/dist-packages:/app:/app/I1

# Por padrão, executa a suíte de validação de contrato de I1
CMD ["python3", "-m", "unittest", "discover", "-s", "I1/tests", "-p", "test_*.py", "-v"]