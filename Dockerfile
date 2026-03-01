# Use the official index-tts-vllm image as base
FROM seastart/index-tts-vllm:latest

# Install runpod SDK
RUN python3 -m pip install --no-cache-dir --break-system-packages runpod

# Follow base image layout: indextts source lives under /app/indextts.
WORKDIR /app
COPY handler.py /app/handler.py

# Set environment variables (can be overridden at runtime)
ENV MODEL_DIR="checkpoints/IndexTTS-2-vLLM"
ENV IS_FP16="true"
ENV GPU_MEMORY_UTILIZATION=0.25
ENV QWENEMO_GPU_MEMORY_UTILIZATION=0.10
ENV PYTHONPATH="${PYTHONPATH}:/app"

# RunPod serverless handler
ENTRYPOINT ["python3", "-u", "/app/handler.py"]
