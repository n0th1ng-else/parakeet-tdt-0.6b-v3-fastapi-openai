host = "0.0.0.0"
port = 5092
threads = 8  # Optimized for 8 P-cores
CHUNK_MINUTE = 1.5  # Target 90-second chunks with intelligent silence-based splitting

# Intelligent chunking configuration
SILENCE_THRESHOLD = "-40dB"  # Silence detection threshold
SILENCE_MIN_DURATION = 0.5  # Minimum silence duration in seconds
SILENCE_SEARCH_WINDOW = 30.0  # Search window in seconds around target split point
SILENCE_DETECT_TIMEOUT = 300  # Timeout for silence detection in seconds
MIN_SPLIT_GAP = 5.0  # Minimum gap between split points to prevent 0-length chunks

import sys

sys.stdout = sys.stderr

import os, sys, json, math, re, threading
import shutil
import uuid
import subprocess
import datetime
import psutil
from typing import List, Tuple, Optional
from werkzeug.utils import secure_filename

import flask
from flask import Flask, request, jsonify, render_template, Response
from waitress import serve
from pathlib import Path

ROOT_DIR = Path(os.getcwd()).as_posix()
os.environ["HF_HOME"] = ROOT_DIR + "/models"
os.environ["HF_HUB_CACHE"] = ROOT_DIR + "/models"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
if sys.platform == "win32":
    os.environ["PATH"] = ROOT_DIR + f";{ROOT_DIR}/ffmpeg;" + os.environ["PATH"]

# Reject requests whose audio exceeds this duration (seconds). Guards
# against malformed inputs that ffmpeg turns into unexpectedly long WAVs
# and would otherwise push the process past its memory budget. Override
# via the MAX_AUDIO_DURATION_SECONDS env var.
MAX_AUDIO_DURATION_SECONDS = float(os.environ.get("MAX_AUDIO_DURATION_SECONDS", 120.0))


# PoC backend tuning (transformers-based models)
HF_MAX_NEW_TOKENS = int(os.environ.get("HF_MAX_NEW_TOKENS", 512))
VOXTRAL_LANGUAGE = os.environ.get("VOXTRAL_LANGUAGE", "en")
# fp32 is usually faster than emulated bf16 on CPU; set to bfloat16 to halve RAM.
VOXTRAL_DTYPE = os.environ.get("VOXTRAL_DTYPE", "float32")
MOONSHINE_LANGUAGE = os.environ.get("MOONSHINE_LANGUAGE", "en")
# e.g. "tiny-streaming", "small-streaming", "medium-streaming"; empty = library default
MOONSHINE_MODEL_ARCH = os.environ.get("MOONSHINE_MODEL_ARCH", "")

# Model configurations. backend "onnx_asr" is the original Parakeet path;
# the other backends are PoC additions loaded lazily on first request.
MODEL_CONFIGS = {
    "parakeet-tdt-0.6b-v3": {
        "backend": "onnx_asr",
        "hf_id": "nemo-parakeet-tdt-0.6b-v3",
        "quantization": "int8",
        "description": "INT8 (fastest)"
    },
    "istupakov/parakeet-tdt-0.6b-v3-onnx": {
        "backend": "onnx_asr",
        "hf_id": "istupakov/parakeet-tdt-0.6b-v3-onnx",
        "quantization": None,
        "description": "FP32"
    },
    "grikdotnet/parakeet-tdt-0.6b-fp16": {
        "backend": "onnx_asr",
        "hf_id": "grikdotnet/parakeet-tdt-0.6b-fp16",
        "quantization": "fp16",
        "description": "FP16"
    },
    "qwen3-asr-0.6b": {
        "backend": "qwen3",
        "hf_id": "Qwen/Qwen3-ASR-0.6B-hf",
        "description": "Qwen3-ASR 0.6B (transformers, fp32 CPU)"
    },
    "moonshine-v2": {
        "backend": "moonshine",
        "language": MOONSHINE_LANGUAGE,
        "model_arch": MOONSHINE_MODEL_ARCH or None,
        "description": "Moonshine v2 (ONNX, CPU)"
    },
    "voxtral-mini": {
        "backend": "voxtral",
        "hf_id": "mistralai/Voxtral-Mini-3B-2507",
        "description": "Voxtral Mini 3B (transformers, CPU)"
    },
}

# Model cache for lazy loading
model_cache = {}

try:
    _ffmpeg_ver = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    print(f"FFmpeg: {_ffmpeg_ver.stdout.splitlines()[0] if _ffmpeg_ver.returncode == 0 else 'not found'}")
except Exception:
    print("FFmpeg: not found")

try:
    print("\nInitializing ONNX Runtime...")
    import onnx_asr
    import onnxruntime as ort
    
    # Detect available providers
    available_providers = ort.get_available_providers()
    print(f"Available providers: {available_providers}")
    
    # Priority: Tensorrt, CUDA, CPU
    providers_to_try = []
    if "TensorrtExecutionProvider" in available_providers:
        providers_to_try.append("TensorrtExecutionProvider")
    if "CUDAExecutionProvider" in available_providers:
        providers_to_try.append("CUDAExecutionProvider")
    providers_to_try.append("CPUExecutionProvider")
    
    print(f"Using providers: {providers_to_try}")

    # Load default INT8 model at startup
    print("\nLoading default Parakeet TDT 0.6B V3 ONNX model with INT8 quantization...")
    
    # Configure session options for optimal CPU performance
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4  # Match Waitress threads
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    default_config = MODEL_CONFIGS["parakeet-tdt-0.6b-v3"]
    asr_model = onnx_asr.load_model(
        default_config["hf_id"],
        quantization=default_config["quantization"],
        providers=providers_to_try,
        sess_options=sess_options,
    ).with_timestamps()
    
    # Cache the default model
    model_cache["parakeet-tdt-0.6b-v3"] = asr_model
    
    print("Default model loaded successfully with CPU optimization!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit()

print("=" * 50)


class HFTranscription:
    """Duck-types the onnx_asr recognize() result (text/tokens/timestamps)
    so the PoC backends plug into the existing chunk pipeline unchanged."""

    def __init__(self, text, timestamps=None, tokens=None):
        self.text = text
        self.timestamps = timestamps or []
        self.tokens = tokens or []


_torch_threads_configured = False


def _configure_torch_threads():
    """Pin torch to physical cores once; override via TORCH_NUM_THREADS."""
    global _torch_threads_configured
    if _torch_threads_configured:
        return
    import torch
    n = int(os.environ.get("TORCH_NUM_THREADS", 0)) or (
        psutil.cpu_count(logical=False) or os.cpu_count() or 4
    )
    torch.set_num_threads(n)
    print(f"torch threads: {n}")
    _torch_threads_configured = True


class Qwen3ASRTranscriber:
    """Qwen/Qwen3-ASR-0.6B-hf via transformers (requires transformers>=5.13)."""

    def __init__(self, hf_id):
        import torch
        from transformers import AutoProcessor, AutoModelForMultimodalLM

        _configure_torch_threads()
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(hf_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(hf_id, dtype=torch.float32)
        self.model.eval()
        # Serialize inference: a single generate() already saturates the CPU
        self._lock = threading.Lock()

    def recognize(self, wav_path):
        inputs = self.processor.apply_transcription_request(audio=wav_path)
        inputs = inputs.to(self.model.device, self.model.dtype)
        with self._lock, self.torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=HF_MAX_NEW_TOKENS)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        text = self.processor.decode(generated_ids, return_format="transcription_only")[0]
        return HFTranscription((text or "").strip())


class MoonshineV2Transcriber:
    """Moonshine v2 via the moonshine-voice package (ONNX runtime, CPU)."""

    def __init__(self, language="en", model_arch=None):
        from moonshine_voice import Transcriber, get_model_for_language, string_to_model_arch

        arch = string_to_model_arch(model_arch) if model_arch else None
        model_path, resolved_arch = get_model_for_language(language, arch)
        print(f"Moonshine model: {model_path} (arch={resolved_arch.name})")
        self.transcriber = Transcriber(model_path=model_path, model_arch=resolved_arch)
        # The ctypes handle is not documented as thread-safe
        self._lock = threading.Lock()

    def recognize(self, wav_path):
        from moonshine_voice import load_wav_file

        audio_data, sample_rate = load_wav_file(wav_path)
        with self._lock:
            transcript = self.transcriber.transcribe_without_streaming(audio_data, sample_rate)
        lines = [ln for ln in transcript.lines if ln.text and ln.text.strip()]
        text = " ".join(ln.text.strip() for ln in lines)
        timestamps = []
        if lines:
            timestamps = [lines[0].start_time, lines[-1].start_time + lines[-1].duration]
        return HFTranscription(text, timestamps=timestamps)


class VoxtralTranscriber:
    """Voxtral via transformers in dedicated transcription mode."""

    def __init__(self, hf_id, language="en", dtype_name="bfloat16"):
        import torch
        from transformers import AutoProcessor, VoxtralForConditionalGeneration

        _configure_torch_threads()
        self.torch = torch
        self.hf_id = hf_id
        self.language = language
        self.dtype = getattr(torch, dtype_name)
        self.processor = AutoProcessor.from_pretrained(hf_id)
        self.model = VoxtralForConditionalGeneration.from_pretrained(hf_id, dtype=self.dtype)
        self.model.eval()
        self._lock = threading.Lock()

    def recognize(self, wav_path):
        inputs = self.processor.apply_transcription_request(
            language=self.language, audio=wav_path, model_id=self.hf_id
        )
        inputs = inputs.to(self.model.device, dtype=self.dtype)
        with self._lock, self.torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=HF_MAX_NEW_TOKENS)
        text = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0]
        return HFTranscription((text or "").strip())


def _load_onnx_asr_model(config):
    """Load a Parakeet ONNX variant with the CPU-tuned session options."""
    import onnxruntime as ort

    # Reuse providers from startup
    available_providers = ort.get_available_providers()
    providers_to_try = []
    if "TensorrtExecutionProvider" in available_providers:
        providers_to_try.append("TensorrtExecutionProvider")
    if "CUDAExecutionProvider" in available_providers:
        providers_to_try.append("CUDAExecutionProvider")
    providers_to_try.append("CPUExecutionProvider")

    # Configure session options
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    return onnx_asr.load_model(
        config["hf_id"],
        quantization=config["quantization"],
        providers=providers_to_try,
        sess_options=sess_options,
    ).with_timestamps()


def get_model(model_name):
    """
    Get or load a model by name with lazy loading and caching.

    Args:
        model_name: Name of the model (key in MODEL_CONFIGS)

    Returns:
        Loaded ASR model instance
    """
    # Default to INT8 if model not found
    if model_name not in MODEL_CONFIGS:
        print(f"⚠️ Unknown model '{model_name}', falling back to default INT8 model")
        model_name = "parakeet-tdt-0.6b-v3"

    # Return cached model if available
    if model_name in model_cache:
        print(f"Using cached model: {model_name}")
        return model_cache[model_name]

    # Load new model
    print(f"Loading model: {model_name}")
    config = MODEL_CONFIGS[model_name]
    backend = config.get("backend", "onnx_asr")

    try:
        if backend == "onnx_asr":
            model = _load_onnx_asr_model(config)
        elif backend == "qwen3":
            model = Qwen3ASRTranscriber(config["hf_id"])
        elif backend == "moonshine":
            model = MoonshineV2Transcriber(
                language=config.get("language", "en"),
                model_arch=config.get("model_arch"),
            )
        elif backend == "voxtral":
            model = VoxtralTranscriber(
                config["hf_id"],
                language=VOXTRAL_LANGUAGE,
                dtype_name=VOXTRAL_DTYPE,
            )
        else:
            raise ValueError(f"Unknown backend '{backend}' for model {model_name}")

        # Cache the loaded model
        model_cache[model_name] = model
        print(f"Model {model_name} loaded successfully")

        return model
    except Exception as e:
        print(f"❌ Failed to load model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        # No silent fallback: substituting another model would make the
        # benchmark results lie about which model produced them.
        raise RuntimeError(f"Failed to load model {model_name}: {e}")


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "temp_uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 2000 * 1024 * 1024

# Progress tracking
progress_tracker = {}


def get_audio_duration(file_path: str) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return float(result.stdout)
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Could not get duration of file '{file_path}': {e}")
        return 0.0


def detect_silence_points(file_path: str, silence_thresh: str = SILENCE_THRESHOLD, 
                          silence_duration: float = SILENCE_MIN_DURATION,
                          total_duration: Optional[float] = None) -> List[Tuple[float, float]]:
    """
    Detect silence points in audio file using ffmpeg's silencedetect filter.
    
    Args:
        file_path: Path to audio file
        silence_thresh: Silence threshold in dB (e.g., "-40dB")
        silence_duration: Minimum silence duration in seconds
        total_duration: Total duration of audio (used to close trailing silence)
        
    Returns:
        List of tuples (silence_start, silence_end) in seconds
    """
    # Validate file exists
    if not os.path.exists(file_path):
        print(f"Error: Audio file '{file_path}' not found for silence detection")
        return []
    
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i", file_path,
        "-af", f"silencedetect=noise={silence_thresh}:d={silence_duration}",
        "-f", "null",
        "-"
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=SILENCE_DETECT_TIMEOUT)
        
        # Parse stderr output for silence intervals
        silence_points = []
        silence_start = None
        
        for line in result.stderr.splitlines():
            if 'silence_start:' in line:
                try:
                    silence_start = float(line.split('silence_start:')[1].split()[0])
                except (ValueError, IndexError):
                    silence_start = None
            elif 'silence_end:' in line and silence_start is not None:
                try:
                    silence_end = float(line.split('silence_end:')[1].split()[0])
                    silence_points.append((silence_start, silence_end))
                    silence_start = None
                except (ValueError, IndexError):
                    pass
        
        # Close trailing silence if audio ended during silence
        if silence_start is not None and total_duration is not None:
            silence_points.append((silence_start, total_duration))
        
        return silence_points
    except subprocess.TimeoutExpired:
        print(f"Timeout: Silence detection exceeded {SILENCE_DETECT_TIMEOUT}s timeout")
        return []
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"Error running FFmpeg for silence detection: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error detecting silence: {e}")
        return []


def find_optimal_split_points(total_duration: float, target_chunk_duration: float, 
                               silence_points: List[Tuple[float, float]], 
                               search_window: float = SILENCE_SEARCH_WINDOW,
                               min_gap: float = MIN_SPLIT_GAP) -> List[float]:
    """
    Find optimal split points based on silence detection.
    
    Args:
        total_duration: Total audio duration in seconds
        target_chunk_duration: Target chunk size in seconds
        silence_points: List of (start, end) tuples for silence periods
        search_window: Search window in seconds around target split point
        min_gap: Minimum gap between split points to prevent 0-length chunks
        
    Returns:
        List of split points in seconds
    """
    if not silence_points or total_duration <= target_chunk_duration:
        return []
    
    split_points = []
    prev = 0.0
    num_chunks = math.ceil(total_duration / target_chunk_duration)
    
    for i in range(1, num_chunks):
        target_time = i * target_chunk_duration
        search_start = max(0.0, target_time - search_window)
        search_end = min(total_duration, target_time + search_window)
        
        # Find silence points that overlap with the search window
        candidates = [
            (start, end) for (start, end) in silence_points
            if start <= search_end and end >= search_start
        ]
        
        chosen = None
        if candidates:
            # Sort candidates by distance from target time
            candidates_sorted = sorted(
                candidates,
                key=lambda silence_range: abs(((silence_range[0] + silence_range[1]) / 2.0) - target_time)
            )
            # Find first candidate that satisfies minimum gap constraint
            for start, end in candidates_sorted:
                split_point = (start + end) / 2.0
                if split_point > prev + min_gap and split_point <= total_duration - min_gap:
                    chosen = split_point
                    break
        
        if chosen is None:
            # Fallback: target time, but enforce monotonicity and bounds
            chosen = max(prev + min_gap, min(target_time, total_duration - min_gap))
            # Ensure chosen doesn't exceed total_duration
            if chosen > total_duration:
                chosen = None  # Skip this split point if not feasible
        
        split_points.append(chosen)
        prev = chosen
    
    # Filter out None values if any splits were skipped
    split_points = [sp for sp in split_points if sp is not None]
    
    return split_points


def format_srt_time(seconds: float) -> str:
    delta = datetime.timedelta(seconds=seconds)
    s = str(delta)
    if "." in s:
        parts = s.split(".")
        integer_part = parts[0]
        fractional_part = parts[1][:3]
    else:
        integer_part = s
        fractional_part = "000"

    if len(integer_part.split(":")) == 2:
        integer_part = "0:" + integer_part

    return f"{integer_part},{fractional_part}"


def segments_to_srt(segments: list) -> str:
    srt_content = []
    for i, segment in enumerate(segments):
        start_time = format_srt_time(segment["start"])
        end_time = format_srt_time(segment["end"])
        text = segment["segment"].strip()

        if text:
            srt_content.append(str(i + 1))
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text)
            srt_content.append("")

    return "\n".join(srt_content)


def segments_to_vtt(segments: list) -> str:
    vtt_content = ["WEBVTT", ""]
    for i, segment in enumerate(segments):
        start_time = format_srt_time(segment["start"]).replace(",", ".")
        end_time = format_srt_time(segment["end"]).replace(",", ".")
        text = segment["segment"].strip()

        if text:
            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(text)
            vtt_content.append("")
    return "\n".join(vtt_content)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/parakeet.png")
def serve_logo():
    return flask.send_file("parakeet.png", mimetype="image/png")


@app.route("/health")
def health():
    available_models = list(MODEL_CONFIGS.keys())
    return jsonify({
        "status": "healthy",
        "models": available_models,
        "default_model": "parakeet-tdt-0.6b-v3",
        "speedup": "20.7x"
    })


@app.route("/docs")
def swagger_ui():
    """Serve Swagger UI"""
    return render_template("swagger.html")


@app.route("/openapi.json")
def openapi_spec():
    """Return OpenAPI Specification"""
    return jsonify({
        "openapi": "3.0.0",
        "info": {
            "title": "Parakeet Transcription API",
            "description": "High-performance ONNX-optimized speech transcription API compatible with OpenAI.",
            "version": "1.0.0"
        },
        "servers": [{"url": "http://100.85.200.51:5092"}],
        "paths": {
            "/v1/audio/transcriptions": {
                "post": {
                    "summary": "Transcribe Audio",
                    "description": "Transcribes audio into the input language. Supports real-time streaming progress.",
                    "operationId": "transcribe_audio",
                    "requestBody": {
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "file": {
                                            "type": "string",
                                            "format": "binary",
                                            "description": "The audio file object (not file name) to transcribe."
                                        },
                                        "model": {
                                            "type": "string",
                                            "default": "parakeet-tdt-0.6b-v3",
                                            "enum": list(MODEL_CONFIGS.keys()),
                                            "description": "Model to use: " + "; ".join(
                                                f"{name} ({cfg['description']})" for name, cfg in MODEL_CONFIGS.items()
                                            )
                                        },
                                        "response_format": {
                                            "type": "string",
                                            "default": "json",
                                            "enum": ["json", "text", "srt", "verbose_json", "vtt"],
                                            "description": "The format of the transcript output."
                                        }
                                    },
                                    "required": ["file"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Successful Response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "text": {"type": "string"}
                                        }
                                    }
                                },
                                "text/plain": {
                                    "schema": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            }
        }
    })


@app.route("/progress/<job_id>")
def get_progress(job_id):
    """Get transcription progress for a job"""
    if job_id in progress_tracker:
        return jsonify(progress_tracker[job_id])
    return jsonify({"status": "not_found"}), 404


@app.route("/status")
def get_status():
    """Get status of the most recent active job"""
    for job_id, progress in progress_tracker.items():
        if progress.get("status") == "processing":
            return jsonify({"job_id": job_id, **progress})
    return jsonify({"status": "idle"})


@app.route("/metrics")
def get_metrics():
    """Get real-time CPU and RAM metrics"""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    return jsonify({
        "cpu_percent": cpu_percent,
        "ram_percent": memory.percent,
        "ram_used_gb": round(memory.used / (1024**3), 2),
        "ram_total_gb": round(memory.total / (1024**3), 2)
    })


@app.route("/v1/audio/transcriptions", methods=["POST"])
def transcribe_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "No file selected"}), 400

    # OpenAI compatible parameters
    model_name = request.form.get("model", "parakeet-tdt-0.6b-v3").lower()
    response_format = request.form.get("response_format", "json")

    print(f"Request Model: {model_name} | Format: {response_format}")
    
    # Validate model and warn if unknown
    original_model_name = model_name
    if model_name not in MODEL_CONFIGS:
        print(f"⚠️ Unknown model '{model_name}' requested, using default")
        model_name = "parakeet-tdt-0.6b-v3"
    
    # Get the appropriate model (with lazy loading)
    model_to_use = get_model(model_name)

    # Legacy support
    if model_name == "parakeet_srt_words":
        pass

    original_filename = secure_filename(file.filename)

    unique_id = str(uuid.uuid4())
    temp_original_path = os.path.join(
        app.config["UPLOAD_FOLDER"], f"{unique_id}_{original_filename}"
    )
    target_wav_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{unique_id}.wav")

    temp_files_to_clean = []

    try:
        file.save(temp_original_path)
        temp_files_to_clean.append(temp_original_path)

        # Pre-convert duration gate: cheap rejection for honest long files
        # before we spend a full decode. The post-convert check still runs to
        # catch containers that under-report duration vs. what ffmpeg decodes.
        source_duration = get_audio_duration(temp_original_path)
        if source_duration > MAX_AUDIO_DURATION_SECONDS:
            print(
                f"[{unique_id}] Rejecting: source audio duration "
                f"{source_duration:.2f}s exceeds MAX_AUDIO_DURATION_SECONDS="
                f"{MAX_AUDIO_DURATION_SECONDS}s"
            )
            return jsonify({
                "error": "Audio too long",
                "duration_seconds": round(source_duration, 2),
                "max_duration_seconds": MAX_AUDIO_DURATION_SECONDS,
            }), 413

        print(
            f"[{unique_id}] Converting '{original_filename}' to standard WAV format..."
        )
        ffmpeg_command = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            temp_original_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            target_wav_path,
        ]
        result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            if "Output file does not contain any stream" in result.stderr:
                return jsonify({"error": "The provided file contains no audio stream to transcribe."}), 400
            return jsonify(
                {"error": "File conversion failed", "details": result.stderr}
            ), 500
        temp_files_to_clean.append(target_wav_path)

        CHUNK_DURATION_SECONDS = CHUNK_MINUTE * 60
        total_duration = get_audio_duration(target_wav_path)
        if total_duration == 0:
            return jsonify({"error": "Cannot process audio with 0 duration"}), 400
        if total_duration > MAX_AUDIO_DURATION_SECONDS:
            print(
                f"[{unique_id}] Rejecting: converted audio duration "
                f"{total_duration:.2f}s exceeds MAX_AUDIO_DURATION_SECONDS="
                f"{MAX_AUDIO_DURATION_SECONDS}s"
            )
            return jsonify({
                "error": "Audio too long after conversion",
                "duration_seconds": round(total_duration, 2),
                "max_duration_seconds": MAX_AUDIO_DURATION_SECONDS,
            }), 413

        # Use intelligent chunking based on silence detection
        chunk_paths = []
        split_points = []
        
        if total_duration > CHUNK_DURATION_SECONDS:
            print(f"[{unique_id}] Detecting silence points for intelligent chunking...")
            silence_points = detect_silence_points(target_wav_path, total_duration=total_duration)
            
            if silence_points:
                print(f"[{unique_id}] Found {len(silence_points)} silence periods")
                split_points = find_optimal_split_points(
                    total_duration, 
                    CHUNK_DURATION_SECONDS, 
                    silence_points,
                    search_window=SILENCE_SEARCH_WINDOW
                )
                print(f"[{unique_id}] Optimal split points: {[f'{sp:.2f}s' for sp in split_points]}")
            else:
                print(f"[{unique_id}] No silence detected, using time-based chunking")
        
        # Create chunks based on split points (or use time-based if no silence found)
        if split_points:
            # Silence-based chunking
            chunk_boundaries = [0.0] + split_points + [total_duration]
            num_chunks = len(chunk_boundaries) - 1
        else:
            # Time-based chunking (fallback)
            num_chunks = math.ceil(total_duration / CHUNK_DURATION_SECONDS)
            chunk_boundaries = [min(i * CHUNK_DURATION_SECONDS, total_duration) for i in range(num_chunks + 1)]
        
        # Initialize progress tracking
        progress_tracker[unique_id] = {
            "status": "processing",
            "current_chunk": 0,
            "total_chunks": num_chunks,
            "progress_percent": 0,
            "partial_text": ""
        }
        
        print(
            f"[{unique_id}] Total duration: {total_duration:.2f}s. Splitting into {num_chunks} chunks."
        )

        if num_chunks > 1:
            for i in range(num_chunks):
                start_time = chunk_boundaries[i]
                duration = chunk_boundaries[i + 1] - start_time
                chunk_path = os.path.join(
                    app.config["UPLOAD_FOLDER"], f"{unique_id}_chunk_{i}.wav"
                )
                chunk_paths.append(chunk_path)
                temp_files_to_clean.append(chunk_path)

                print(f"[{unique_id}] Creating chunk {i + 1}/{num_chunks} ({start_time:.2f}s - {chunk_boundaries[i+1]:.2f}s)...")
                chunk_command = [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-ss",
                    str(start_time),
                    "-t",
                    str(duration),
                    "-i",
                    target_wav_path,
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    chunk_path,
                ]
                result = subprocess.run(chunk_command, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Warning: Chunk extraction failed: {result.stderr}")
        else:
            chunk_paths.append(target_wav_path)

        all_segments = []
        all_words = []
        cumulative_time_offset = 0.0
        
        # Store chunk durations for offset calculation
        chunk_durations = []
        if num_chunks > 1:
            for i in range(num_chunks):
                duration = chunk_boundaries[i + 1] - chunk_boundaries[i]
                chunk_durations.append(duration)
        else:
            chunk_durations.append(total_duration)

        def clean_text(text):
            """Clean up spacing artifacts from token joining"""
            if not text:
                return ""
            # Handle potential SentencePiece underline
            text = text.replace("\u2581", " ")
            text = text.strip()
            # Collapse multiple spaces
            text = re.sub(r"\s+", " ", text)
            # Standard cleaning
            text = text.replace(" '", "'")
            return text

        for i, chunk_path in enumerate(chunk_paths):
            progress_tracker[unique_id].update({
                "current_chunk": i + 1,
                "progress_percent": int((i + 1) / num_chunks * 100)
            })
            print(f"[{unique_id}] Transcribing chunk {i + 1}/{num_chunks}...")

            result = model_to_use.recognize(chunk_path)

            if result and result.text:
                start_time = result.timestamps[0] if result.timestamps else 0
                end_time = (
                    result.timestamps[-1]
                    if len(result.timestamps) > 1
                    else start_time + 0.1
                )

                cleaned_text = clean_text(result.text)

                segment = {
                    "start": start_time + cumulative_time_offset,
                    "end": end_time + cumulative_time_offset,
                    "segment": cleaned_text,
                }
                all_segments.append(segment)
                
                # Update partial text for real-time streaming
                progress_tracker[unique_id]["partial_text"] += cleaned_text + " "

                for j, (token, timestamp) in enumerate(
                    zip(result.tokens, result.timestamps)
                ):
                    if j < len(result.timestamps) - 1:
                        word_end = result.timestamps[j + 1]
                    else:
                        word_end = end_time

                    # Clean tokens too
                    clean_token = token.replace("\u2581", " ").strip()
                    word = {
                        "start": timestamp + cumulative_time_offset,
                        "end": word_end + cumulative_time_offset,
                        "word": clean_token,
                    }
                    all_words.append(word)

            # Use planned chunk duration instead of ffprobe
            cumulative_time_offset += chunk_durations[i]

        print(f"[{unique_id}] All chunks transcribed, merging results.")
        
        # Update progress to complete
        progress_tracker[unique_id]["status"] = "complete"
        progress_tracker[unique_id]["progress_percent"] = 100

        if not all_segments:
            # Return empty structure if nothing found, consistent with failures or silence?
            # OpenAI sometimes returns empty json text.
            pass

        # Formatting Output
        full_text = " ".join([seg["segment"] for seg in all_segments])

        if response_format == "srt" or model_name == "parakeet_srt_words":
            srt_output = segments_to_srt(all_segments)
            if model_name == "parakeet_srt_words":
                json_str_list = [
                    {"start": it["start"], "end": it["end"], "word": it["word"]}
                    for it in all_words
                ]
                srt_output += "----..----" + json.dumps(json_str_list)
            return Response(srt_output, mimetype="text/plain")

        elif response_format == "vtt":
            return Response(segments_to_vtt(all_segments), mimetype="text/plain")

        elif response_format == "text":
            return Response(full_text, mimetype="text/plain")

        elif response_format == "verbose_json":
            # Minimal verbose_json structure
            return jsonify(
                {
                    "task": "transcribe",
                    "language": "english",  # detection not implemented here, hardcoded or param?
                    "duration": total_duration,
                    "text": full_text,
                    "segments": [
                        {
                            "id": idx,
                            "seek": 0,
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": seg["segment"],
                            "tokens": [],  # Populate if needed
                            "temperature": 0.0,
                            "avg_logprob": 0.0,
                            "compression_ratio": 0.0,
                            "no_speech_prob": 0.0,
                        }
                        for idx, seg in enumerate(all_segments)
                    ],
                }
            )

        else:
            # Default JSON
            response = jsonify({"text": full_text})
            response.headers['X-Job-ID'] = unique_id
            return response

    except Exception as e:
        print(f"A serious error occurred during processing: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": str(e)}), 500
    finally:
        print(f"[{unique_id}] Cleaning up temporary files...")
        for f_path in temp_files_to_clean:
            if os.path.exists(f_path):
                os.remove(f_path)
        print(f"[{unique_id}] Temporary files cleaned.")


def openweb():
    import webbrowser, time

    time.sleep(5)
    webbrowser.open_new_tab(f"http://127.0.0.1:{port}")


if __name__ == "__main__":
    print(f"Starting server...")
    print(f"Web interface: http://127.0.0.1:{port}")
    print(f"API Endpoint: POST http://{host}:{port}/v1/audio/transcriptions")
    print(f"Running with {threads} threads.")
    print(f"Max audio duration: {MAX_AUDIO_DURATION_SECONDS}s")
    print(f"Starting web browser thread...")
    threading.Thread(target=openweb).start()
    print(f"Starting waitress server...")
    serve(app, host=host, port=port, threads=threads)
    print(f"Server started!")
