import os
import sys
from pathlib import Path
import importlib.util

import runpod
import base64
import io
import soundfile as sf
import requests
import uuid

def log_runtime_snapshot():
    print(f"INFO: handler_file={Path(__file__).resolve()}")
    print(f"INFO: executable={sys.executable}")
    print(f"INFO: cwd={os.getcwd()}")
    print(f"INFO: env.PYTHONPATH={os.getenv('PYTHONPATH', '')}")
    print(f"INFO: sys.path={sys.path}")

    inspect_dirs = [
        Path("/"),
        Path("/app"),
        Path("/worker"),
        Path("/opt/index-tts"),
        Path("/opt/index-tts/indextts"),
    ]
    for d in inspect_dirs:
        exists = d.exists()
        is_dir = d.is_dir()
        sample = []
        if is_dir:
            try:
                sample = sorted([p.name for p in d.iterdir()])[:12]
            except Exception as e:
                sample = [f"<list_failed:{e}>"]
        print(f"INFO: dir={d} exists={exists} is_dir={is_dir} sample={sample}")

def bootstrap_indextts_import():
    existing = importlib.util.find_spec("indextts")
    if existing is not None:
        return f"installed:{existing.origin}"

    candidates = [
        Path("/opt/index-tts"),
        Path("/app"),
        Path("/workspace"),
        Path("/src"),
    ]
    checked = []
    for root in candidates:
        has_module = (root / "indextts").is_dir()
        checked.append((str(root), has_module))
        if has_module:
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            probe = importlib.util.find_spec("indextts")
            if probe is not None:
                return f"path:{root_str}"

    details = ", ".join([f"{root}={exists}" for root, exists in checked])
    raise ModuleNotFoundError(
        "No module named 'indextts'. Checked package import and source dirs: "
        f"{details}. sys.path={sys.path}"
    )

def resolve_model_dir():
    env_model_dir = os.getenv("MODEL_DIR", "").strip()
    if env_model_dir:
        return env_model_dir

    candidates = [
        "/app/checkpoints/IndexTTS-2-vLLM",
        "/workspace/checkpoints/IndexTTS-2-vLLM",
        "/opt/index-tts/checkpoints/IndexTTS-2-vLLM",
        "checkpoints/IndexTTS-2-vLLM",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    return "checkpoints/IndexTTS-2-vLLM"

log_runtime_snapshot()
INDEXTTS_SOURCE = bootstrap_indextts_import()

try:
    from indextts.infer_vllm_v2 import IndexTTS2
except ImportError as e:
    print(f"ERROR: Failed to import indextts: {e}")
    print(f"DEBUG: CWD: {os.getcwd()}")
    print(f"DEBUG: PYTHONPATH: {os.getenv('PYTHONPATH', '')}")
    print(f"DEBUG: sys.path: {sys.path}")
    raise

# Initialize the model outside the handler for warm starts
model_dir = resolve_model_dir()
is_fp16 = os.getenv("IS_FP16", "true").lower() == "true"
gpu_memory_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.25"))
qwenemo_gpu_memory_utilization = float(os.getenv("QWENEMO_GPU_MEMORY_UTILIZATION", "0.10"))

print(f"INFO: indextts source: {INDEXTTS_SOURCE}")
print(f"INFO: model_dir: {model_dir}")

tts = IndexTTS2(
    model_dir=model_dir,
    is_fp16=is_fp16,
    gpu_memory_utilization=gpu_memory_utilization,
    qwenemo_gpu_memory_utilization=qwenemo_gpu_memory_utilization,
)

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    return False

def process_audio_input(audio_input, temp_dir="/tmp"):
    if not audio_input:
        return None
        
    if audio_input.startswith(("http://", "https://")):
        local_path = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")
        if download_file(audio_input, local_path):
            return local_path
        return None
        
    if audio_input.startswith("data:audio/"):
        try:
            _, encoded = audio_input.split(",", 1)
            decoded = base64.b64decode(encoded)
            local_path = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")
            with open(local_path, "wb") as f:
                f.write(decoded)
            return local_path
        except Exception:
            return None
            
    try:
        decoded = base64.b64decode(audio_input)
        local_path = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")
        with open(local_path, "wb") as f:
            f.write(decoded)
        return local_path
    except Exception:
        return audio_input

async def handler(job):
    job_input = job['input']
    
    text = job_input.get("text")
    spk_audio_input = job_input.get("spk_audio_path")
    emo_control_method = job_input.get("emo_control_method", 0)
    emo_ref_input = job_input.get("emo_ref_path", None)
    emo_weight = job_input.get("emo_weight", 1.0)
    emo_vec = job_input.get("emo_vec", None)
    emo_text = job_input.get("emo_text", None)
    emo_random = job_input.get("emo_random", False)
    max_text_tokens_per_sentence = job_input.get("max_text_tokens_per_sentence", 120)

    spk_audio_path = process_audio_input(spk_audio_input)
    emo_ref_path = process_audio_input(emo_ref_input)

    if emo_control_method == 0:
        emo_ref_path = None
        emo_weight = 1.0
        vec = None
    elif emo_control_method == 2:
        vec = emo_vec
        if sum(vec) > 1.5:
             return {"error": "情感向量之和不能超过1.5"}
    else:
        vec = None

    try:
        sr, wav = await tts.infer(
            spk_audio_prompt=spk_audio_path, 
            text=text,
            output_path=None,
            emo_audio_prompt=emo_ref_path, 
            emo_alpha=emo_weight,
            emo_vector=vec,
            use_emo_text=(emo_control_method==3), 
            emo_text=emo_text,
            use_random=emo_random,
            max_text_tokens_per_sentence=int(max_text_tokens_per_sentence)
        )
        
        for p in [spk_audio_path, emo_ref_path]:
            if p and p.startswith("/tmp/") and os.path.exists(p):
                try: os.remove(p)
                except: pass

        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()
        
        audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
        
        return {
            "status": "success",
            "audio_base64": audio_base64,
            "format": "wav",
            "sample_rate": sr
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})
