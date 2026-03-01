# Use the official index-tts-vllm image as base
FROM seastart/index-tts-vllm:latest

# Install runpod SDK
RUN python3 -m pip install --no-cache-dir --break-system-packages runpod

# Probe common locations in the base image and keep a stable copy of source code.
RUN python3 - <<'PY'
import importlib.util
import shutil
from pathlib import Path

candidates = [Path("/app"), Path("/workspace"), Path("/src"), Path("/opt")]
stable_root = Path("/opt/index-tts")
stable_root.mkdir(parents=True, exist_ok=True)

copied = False
for root in candidates:
    src = root / "indextts"
    if src.is_dir():
        target = stable_root / "indextts"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(src, target, symlinks=True)
        tools_src = root / "tools"
        if tools_src.is_dir():
            tools_dst = stable_root / "tools"
            if tools_dst.exists():
                shutil.rmtree(tools_dst)
            shutil.copytree(tools_src, tools_dst, symlinks=True)
        print(f"INFO: copied indextts from {src} to {target}")
        copied = True
        break

spec = importlib.util.find_spec("indextts")
if spec is not None:
    print(f"INFO: indextts is importable at build time from {spec.origin}")

if not copied and spec is None:
    raise SystemExit(
        "ERROR: indextts was neither found as source directory nor as installed package "
        "in base image."
    )
PY

# Keep runtime code away from /app in case serverless runtime overlays /app.
WORKDIR /worker
COPY handler.py /worker/handler.py
COPY handler.py /handler.py
COPY handler.py /app/handler.py

# Set environment variables (can be overridden at runtime)
ENV MODEL_DIR=""
ENV IS_FP16="true"
ENV GPU_MEMORY_UTILIZATION=0.25
ENV QWENEMO_GPU_MEMORY_UTILIZATION=0.10
ENV PYTHONPATH="/opt/index-tts:/worker:/app:/"

# RunPod serverless handler
ENTRYPOINT ["python3", "-u", "/worker/handler.py"]
