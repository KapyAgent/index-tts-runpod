# Use the official index-tts-vllm image as base
FROM seastart/index-tts-vllm:latest

# Install runpod SDK
RUN pip install runpod

# Copy our handler
COPY handler.py .

# Set environment variables (can be overridden at runtime)
ENV MODEL_DIR="checkpoints/IndexTTS-2-vLLM"
ENV IS_FP16="true"
ENV GPU_MEMORY_UTILIZATION=0.25
ENV QWENEMO_GPU_MEMORY_UTILIZATION=0.10

# Override entrypoint to run the handler
ENTRYPOINT ["python", "-u", "handler.py"]
