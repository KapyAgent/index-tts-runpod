# Use the official index-tts-vllm image as base
FROM seastart/index-tts-vllm:latest

# Install runpod SDK
RUN pip install --no-cache-dir --break-system-packages runpod

# Keep a stable copy of source code outside /app so imports do not depend on cwd.
RUN mkdir -p /opt/index-tts && \
    cp -a /app/indextts /opt/index-tts/indextts && \
    cp -a /app/tools /opt/index-tts/tools

# Put worker code in its own directory.
WORKDIR /worker
COPY handler.py /worker/handler.py

# Set environment variables (can be overridden at runtime)
ENV MODEL_DIR="checkpoints/IndexTTS-2-vLLM"
ENV IS_FP16="true"
ENV GPU_MEMORY_UTILIZATION=0.25
ENV QWENEMO_GPU_MEMORY_UTILIZATION=0.10
ENV PYTHONPATH="/opt/index-tts:/app:/worker"

# RunPod serverless handler
ENTRYPOINT ["python", "-u", "/worker/handler.py"]
