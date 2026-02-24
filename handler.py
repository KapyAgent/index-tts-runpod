import runpod
import base64
import io
import soundfile as sf
import os
from indextts.infer_vllm_v2 import IndexTTS2

# Initialize the model outside the handler for warm starts
model_dir = os.getenv("MODEL_DIR", "checkpoints/IndexTTS-2-vLLM")
is_fp16 = os.getenv("IS_FP16", "true").lower() == "true"
gpu_memory_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.25"))
qwenemo_gpu_memory_utilization = float(os.getenv("QWENEMO_GPU_MEMORY_UTILIZATION", "0.10"))

tts = IndexTTS2(
    model_dir=model_dir,
    is_fp16=is_fp16,
    gpu_memory_utilization=gpu_memory_utilization,
    qwenemo_gpu_memory_utilization=qwenemo_gpu_memory_utilization,
)

async def handler(job):
    """
    The handler function that will be called by RunPod.
    job['input'] contains the parameters sent by the user.
    """
    job_input = job['input']
    
    text = job_input.get("text")
    spk_audio_path = job_input.get("spk_audio_path")
    emo_control_method = job_input.get("emo_control_method", 0)
    emo_ref_path = job_input.get("emo_ref_path", None)
    emo_weight = job_input.get("emo_weight", 1.0)
    emo_vec = job_input.get("emo_vec", None)
    emo_text = job_input.get("emo_text", None)
    emo_random = job_input.get("emo_random", False)
    max_text_tokens_per_sentence = job_input.get("max_text_tokens_per_sentence", 120)

    # Process emo_control_method logic similar to api_server_v2.py
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
