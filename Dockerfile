FROM python:3.11-slim

WORKDIR /app

# Dependências de sistema mínimas para OpenCV headless e librdkafka
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências leves para execução, validação e Kafka
RUN pip install --no-cache-dir \
    numpy \
    opencv-python-headless \
    pyyaml \
    confluent-kafka \
    ultralytics \
    onnxruntime

COPY B5/ /app/B5/
COPY I1/ /app/I1/
COPY schemas/ /app/schemas/

ENV PYTHONPATH=/app:/app/I1
ENV ODS_KAFKA_BROKER=34.66.235.91:9093

# Por padrão, executa a suíte de validação de contrato de I1
CMD ["python3", "I1/tests/test_i1_sprint3.py"]