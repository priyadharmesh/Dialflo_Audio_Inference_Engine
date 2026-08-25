"""
Dialflo Voice Demographic Service - Main API Module

This module exposes the FastAPI endpoints for audio analysis.

CRITICAL DESIGN DECISIONS:
1. CPU-bound work runs in ThreadPoolExecutor (doesn't block event loop)
2. File uploads read directly into memory (no disk spillover)
3. Strict file size limits enforced BEFORE reading
"""

import asyncio
import io
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app.exceptions import (
    AudioDecoderError,
    AudioTooLongError,
    AudioTooShortError,
    CorruptedAudioError,
    EmptyAudioError,
    UnsupportedFormatError,
)
from app.services.audio_decoder import decode_audio_bytes_detailed
from app.services.quality_analyzer import (
    analyze_quality,
    AudioQuality,
    get_confidence_multiplier
)
from app.services.inference_engine import init_engine, get_engine


# =============================================================================
# CONFIGURATION
# =============================================================================

# Maximum file size: 10 MB (enough for 60+ seconds of most audio formats)
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# Thread pool for CPU-bound operations (prevents event loop blocking)
# Rule of thumb: workers = number of CPU cores for CPU-bound work
# Note: Created lazily to handle test scenarios properly
_cpu_executor: Optional[ThreadPoolExecutor] = None


def get_cpu_executor() -> ThreadPoolExecutor:
    """Get or create the CPU thread pool executor."""
    global _cpu_executor
    if _cpu_executor is None or _cpu_executor._shutdown:
        _cpu_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cpu_worker")
    return _cpu_executor


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("dialflo-backend")


# =============================================================================
# APPLICATION LIFECYCLE
# =============================================================================

def _warmup_full_pipeline():
    """
    Pre-load AND EXECUTE the full pipeline to avoid cold-start latency.

    Cold start causes:
    - PyAV codec initialization: ~500ms on first decode
    - scipy filter compilation: ~100ms on first filter
    - numpy JIT warmup: ~50ms on first array operations

    This warmup runs the ENTIRE pipeline with synthetic audio so that
    the first real request doesn't pay the initialization cost.

    Without warmup: First request ~1500ms
    With warmup: First request ~200ms
    """
    import io
    import wave
    import numpy as np

    start = time.perf_counter()
    logger.info("Running full pipeline warmup...")

    try:
        # === STEP 1: Create synthetic WAV audio (1 second) ===
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Generate speech-like audio (150Hz fundamental with harmonics)
        audio = np.zeros_like(t)
        for harmonic in range(1, 5):
            audio += (0.3 / harmonic) * np.sin(2 * np.pi * 150 * harmonic * t)
        audio = (audio * 32767).astype(np.int16)

        # Write to WAV in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio.tobytes())
        wav_bytes = wav_buffer.getvalue()

        # === STEP 2: Run decoder (warms up soundfile) ===
        from app.services.audio_decoder import decode_audio_bytes_detailed
        decode_result = decode_audio_bytes_detailed(wav_bytes, validate=False)
        logger.debug(f"Warmup decode: {decode_result.decoding_ms:.0f}ms")

        # === STEP 3: Run quality analyzer (warms up scipy filters) ===
        from app.services.quality_analyzer import analyze_quality
        quality_result = analyze_quality(decode_result.audio, decode_result.sample_rate)
        logger.debug(f"Warmup quality: {quality_result.analysis_ms:.0f}ms")

        # === STEP 4: Run inference engine (warms up pitch detection) ===
        from app.services.inference_engine import get_engine
        engine = get_engine()
        inference_result = engine.predict(decode_result.audio, decode_result.sample_rate)
        logger.debug(f"Warmup inference: {inference_result.inference_ms:.0f}ms")

        # === STEP 5: Warm up PyAV codecs (MP3/AAC are common) ===
        import av
        # Just importing av initializes FFmpeg
        # Creating a dummy container forces codec registration
        try:
            dummy_buffer = io.BytesIO(b'\x00' * 100)
            av.open(dummy_buffer, mode='r')
        except Exception:
            pass  # Expected to fail, but codecs are now loaded

    except Exception as e:
        logger.warning(f"Warmup encountered error (non-fatal): {e}")

    elapsed = (time.perf_counter() - start) * 1000
    logger.info(f"Full pipeline warmup completed in {elapsed:.0f}ms")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Startup: Load ML models, warmup decoders
    Shutdown: Clean up resources (thread pool)
    """
    # === STARTUP ===
    logger.info("Starting Dialflo Voice Demographic Service...")

    # Load ML inference engine FIRST (needed for warmup)
    logger.info("Loading ML inference engine...")
    try:
        init_engine()
        logger.info("ML inference engine loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load ML engine: {e}")
        logger.warning("Service will run but predictions will be unavailable")

    # Full pipeline warmup (eliminates cold-start latency)
    _warmup_full_pipeline()

    logger.info("Service ready to accept requests")

    yield  # Application runs here

    # === SHUTDOWN ===
    logger.info("Shutting down service...")

    # Gracefully shutdown thread pool (if it was created)
    global _cpu_executor
    if _cpu_executor is not None and not _cpu_executor._shutdown:
        _cpu_executor.shutdown(wait=True, cancel_futures=False)
        _cpu_executor = None
        logger.info("Thread pool shut down")


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Dialflo Voice Demographic Service",
    description="Real-time gender and age inference from voice audio",
    version="1.0.0",
    lifespan=lifespan
)


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

@app.exception_handler(AudioDecoderError)
async def audio_decoder_error_handler(request, exc: AudioDecoderError):
    """Handle all audio decoding errors with appropriate HTTP status codes."""

    # Map exception types to HTTP status codes
    if isinstance(exc, EmptyAudioError):
        status_code = 400
    elif isinstance(exc, (AudioTooShortError, AudioTooLongError)):
        status_code = 422
    elif isinstance(exc, UnsupportedFormatError):
        status_code = 415  # Unsupported Media Type
    elif isinstance(exc, CorruptedAudioError):
        status_code = 422
    else:
        status_code = 500

    return JSONResponse(
        status_code=status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details
        }
    )


# =============================================================================
# MEMORY-SAFE FILE UPLOAD HELPER
# =============================================================================

async def read_upload_to_memory(
    file: UploadFile,
    max_size: int = MAX_FILE_SIZE_BYTES
) -> bytes:
    """
    Read uploaded file directly into memory with strict size limits.

    This bypasses Starlette's SpooledTemporaryFile to guarantee:
    - NO disk writes (PII compliance)
    - Strict size limits enforced DURING read (not after)
    - Memory-efficient chunked reading

    Args:
        file: FastAPI UploadFile object
        max_size: Maximum allowed file size in bytes

    Returns:
        File contents as bytes

    Raises:
        HTTPException 413 if file exceeds max_size
    """
    chunks = []
    total_size = 0
    chunk_size = 64 * 1024  # 64 KB chunks

    # Read in chunks to enforce size limit during upload
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break

        total_size += len(chunk)

        # Enforce size limit DURING read (fail fast)
        if total_size > max_size:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "FileTooLarge",
                    "message": f"File exceeds maximum size of {max_size // (1024*1024)} MB",
                    "max_bytes": max_size,
                    "received_bytes": total_size
                }
            )

        chunks.append(chunk)

    return b''.join(chunks)


# =============================================================================
# CPU-BOUND WORK WRAPPER
# =============================================================================

async def run_in_thread_pool(func, *args):
    """
    Run a synchronous CPU-bound function in the thread pool.

    This prevents blocking the asyncio event loop, allowing the server
    to handle other requests while CPU work is processing.

    Args:
        func: Synchronous function to run
        *args: Arguments to pass to the function

    Returns:
        Result of the function
    """
    loop = asyncio.get_event_loop()
    executor = get_cpu_executor()
    return await loop.run_in_executor(executor, func, *args)


# =============================================================================
# SYNCHRONOUS PROCESSING FUNCTION (runs in thread pool)
# =============================================================================

def _process_audio_sync(audio_bytes: bytes, contact_id: str):
    """
    Synchronous audio processing pipeline.

    This function contains all CPU-bound work and runs in the thread pool.
    Keeping it synchronous makes it easier to reason about and debug.

    Args:
        audio_bytes: Raw audio file bytes
        contact_id: Request tracking ID

    Returns:
        dict with processing results
    """
    result = {
        "decoding_result": None,
        "quality_result": None,
        "inference_result": None,
        "error": None
    }

    try:
        # STEP 1: Decode audio
        result["decoding_result"] = decode_audio_bytes_detailed(audio_bytes)

        # STEP 2: Quality analysis
        result["quality_result"] = analyze_quality(
            result["decoding_result"].audio,
            result["decoding_result"].sample_rate
        )

        # STEP 3: ML inference (skip if quality insufficient)
        if result["quality_result"].quality != AudioQuality.INSUFFICIENT:
            engine = get_engine()
            result["inference_result"] = engine.predict(
                result["decoding_result"].audio,
                result["decoding_result"].sample_rate
            )

    except Exception as e:
        result["error"] = e

    return result


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy", "service": "dialflo-voice-demographic"}


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyze audio to infer speaker demographics (gender and age bracket).

    This endpoint accepts an audio file upload and returns:
    - Gender prediction with confidence score
    - Age bracket prediction with confidence score
    - Audio quality assessment
    - Processing time metrics

    **Supported Formats:** WAV, MP3, OGG, FLAC, WebM, AAC

    **File Size Limit:** 10 MB maximum

    **Privacy:** Audio is processed entirely in-memory and NEVER written to disk.

    **Performance:** CPU-bound work runs in thread pool to prevent blocking.
    """
    request_start = time.perf_counter()
    contact_id = str(uuid.uuid4())

    logger.info(
        f"[{contact_id}] Received request: "
        f"filename={file.filename}, content_type={file.content_type}"
    )

    # =================================================================
    # STEP 1: READ FILE INTO MEMORY (with size limit enforcement)
    # =================================================================
    # This uses our custom reader that:
    # - Never writes to disk (bypasses SpooledTemporaryFile)
    # - Enforces size limit during read (fails fast)
    # - Reads in chunks (memory efficient)

    audio_bytes = await read_upload_to_memory(file, MAX_FILE_SIZE_BYTES)
    payload_size = len(audio_bytes)

    logger.info(f"[{contact_id}] Read {payload_size} bytes into memory")

    if payload_size == 0:
        raise EmptyAudioError("Received empty audio payload", num_samples=0)

    # =================================================================
    # STEP 2: PROCESS IN THREAD POOL (non-blocking)
    # =================================================================
    # All CPU-bound work (decoding, quality analysis, inference)
    # runs in a separate thread to keep the event loop responsive.

    processing_result = await run_in_thread_pool(
        _process_audio_sync,
        audio_bytes,
        contact_id
    )

    # Re-raise any exception from the processing thread
    if processing_result["error"] is not None:
        raise processing_result["error"]

    decoding_result = processing_result["decoding_result"]
    quality_result = processing_result["quality_result"]
    inference_result = processing_result["inference_result"]

    logger.info(
        f"[{contact_id}] Decoded: format={decoding_result.original_format.value}, "
        f"duration={decoding_result.duration_seconds:.2f}s, "
        f"decode_time={decoding_result.decoding_ms:.1f}ms"
    )

    logger.info(
        f"[{contact_id}] Quality: {quality_result.quality.value}, "
        f"SNR={quality_result.snr_estimate_db:.1f}dB, "
        f"speech_ratio={quality_result.speech_ratio:.2f}"
    )

    # =================================================================
    # STEP 3: BUILD RESPONSE
    # =================================================================

    # Handle insufficient quality case
    if quality_result.quality == AudioQuality.INSUFFICIENT:
        total_processing_ms = int((time.perf_counter() - request_start) * 1000)
        return {
            "contact_id": contact_id,
            "gender": {"prediction": "unknown", "confidence": 0.0},
            "age_bracket": {"prediction": "unknown", "confidence": 0.0},
            "processing_ms": total_processing_ms,
            "audio_quality": quality_result.quality.value,
            "_debug": {
                "original_format": decoding_result.original_format.value,
                "duration_seconds": round(decoding_result.duration_seconds, 2),
                "decoding_ms": round(decoding_result.decoding_ms, 1),
                "quality_analysis_ms": round(quality_result.analysis_ms, 1),
                "reason": "Audio quality insufficient for inference",
                "snr_db": round(quality_result.snr_estimate_db, 1),
                "speech_ratio": round(quality_result.speech_ratio, 2)
            }
        }

    # Apply confidence multiplier based on audio quality
    confidence_multiplier = get_confidence_multiplier(quality_result.quality)

    if inference_result:
        adjusted_gender_conf = inference_result.gender_confidence * confidence_multiplier
        adjusted_age_conf = inference_result.age_confidence * confidence_multiplier

        gender_prediction = {
            "prediction": inference_result.gender_prediction,
            "confidence": round(adjusted_gender_conf, 2)
        }
        age_prediction = {
            "prediction": inference_result.age_bracket,
            "confidence": round(adjusted_age_conf, 2)
        }
        inference_ms = inference_result.inference_ms
    else:
        gender_prediction = {"prediction": "unknown", "confidence": 0.0}
        age_prediction = {"prediction": "unknown", "confidence": 0.0}
        inference_ms = 0

    total_processing_ms = int((time.perf_counter() - request_start) * 1000)

    response = {
        "contact_id": contact_id,
        "gender": gender_prediction,
        "age_bracket": age_prediction,
        "processing_ms": total_processing_ms,
        "audio_quality": quality_result.quality.value,
        # Debug info (can be removed in production)
        "_debug": {
            "original_format": decoding_result.original_format.value,
            "original_sample_rate": decoding_result.original_sample_rate,
            "duration_seconds": round(decoding_result.duration_seconds, 2),
            "decoding_ms": round(decoding_result.decoding_ms, 1),
            "quality_analysis_ms": round(quality_result.analysis_ms, 1),
            "inference_ms": round(inference_ms, 1),
            "snr_db": round(quality_result.snr_estimate_db, 1),
            "speech_ratio": round(quality_result.speech_ratio, 2),
            "confidence_multiplier": confidence_multiplier,
            "num_samples": len(decoding_result.audio),
        }
    }

    logger.info(f"[{contact_id}] Request completed in {total_processing_ms}ms")

    return response


# =============================================================================
# WEBSOCKET STREAMING ENDPOINT
# =============================================================================

# Minimum audio duration before first prediction (seconds)
MIN_AUDIO_FOR_PREDICTION = 2.0

# How often to send updates (seconds of new audio)
UPDATE_INTERVAL_SECONDS = 2.0


def _fix_wav_header_length(audio_bytes: bytes) -> bytes:
    """
    Fix WAV header to match actual data length.

    Problem: WAV header contains file length. When streaming, we have
    partial data but header says "file is X bytes". Decoder waits forever.

    Solution: Rewrite the length fields in the header to match actual data.

    WAV format:
    - Bytes 0-3: "RIFF"
    - Bytes 4-7: File size - 8 (little-endian uint32)
    - Bytes 8-11: "WAVE"
    - ...
    - "data" chunk has size at offset+4

    Returns: Fixed audio bytes (or original if not WAV)
    """
    # Check if it's a WAV file
    if len(audio_bytes) < 44:  # Minimum WAV header size
        return audio_bytes

    if audio_bytes[:4] != b'RIFF' or audio_bytes[8:12] != b'WAVE':
        return audio_bytes  # Not a WAV file

    # Convert to bytearray for modification
    data = bytearray(audio_bytes)

    # Fix RIFF chunk size (bytes 4-7): total file size - 8
    riff_size = len(data) - 8
    data[4:8] = riff_size.to_bytes(4, 'little')

    # Find and fix "data" chunk size
    # The data chunk starts after the header, search for "data" marker
    pos = 12  # Start after "WAVE"
    while pos < len(data) - 8:
        chunk_id = bytes(data[pos:pos + 4])
        chunk_size = int.from_bytes(data[pos + 4:pos + 8], 'little')

        if chunk_id == b'data':
            # Fix data chunk size to match remaining bytes
            actual_data_size = len(data) - pos - 8
            data[pos + 4:pos + 8] = actual_data_size.to_bytes(4, 'little')
            break

        # Move to next chunk
        pos += 8 + chunk_size

        # Handle odd chunk sizes (WAV chunks are word-aligned)
        if chunk_size % 2 == 1:
            pos += 1

    return bytes(data)


def _process_streaming_audio(audio_bytes: bytes, session_id: str) -> dict:
    """
    Process accumulated audio for streaming prediction.

    Reuses the same pipeline as POST /analyze:
    - decode_audio_bytes_detailed()
    - analyze_quality()
    - engine.predict()

    Returns a dict suitable for WebSocket JSON response.

    NOTE: For WAV files, we fix the header length to match actual data
    to avoid decoder issues with incomplete files.
    """
    try:
        # Fix WAV header if needed (WAV files have length in header)
        logger.debug(f"[{session_id}] Fixing WAV header for {len(audio_bytes)} bytes")
        audio_bytes = _fix_wav_header_length(audio_bytes)

        # Decode accumulated audio
        logger.debug(f"[{session_id}] Starting decode...")
        decoding_result = decode_audio_bytes_detailed(audio_bytes, validate=False)
        logger.debug(f"[{session_id}] Decode complete: {decoding_result.duration_seconds:.2f}s")

        # Check minimum duration
        if decoding_result.duration_seconds < MIN_AUDIO_FOR_PREDICTION:
            return {
                "type": "waiting",
                "session_id": session_id,
                "message": f"Need at least {MIN_AUDIO_FOR_PREDICTION}s of audio",
                "duration_so_far": round(decoding_result.duration_seconds, 2)
            }

        # Quality analysis
        quality_result = analyze_quality(
            decoding_result.audio,
            decoding_result.sample_rate
        )

        # Skip inference if quality is insufficient
        if quality_result.quality == AudioQuality.INSUFFICIENT:
            return {
                "type": "partial",
                "session_id": session_id,
                "duration_seconds": round(decoding_result.duration_seconds, 2),
                "audio_quality": quality_result.quality.value,
                "gender": {"prediction": "unknown", "confidence": 0.0},
                "age_bracket": {"prediction": "unknown", "confidence": 0.0},
                "message": "Audio quality insufficient"
            }

        # Run inference
        engine = get_engine()
        inference_result = engine.predict(
            decoding_result.audio,
            decoding_result.sample_rate
        )

        # Apply confidence multiplier
        confidence_multiplier = get_confidence_multiplier(quality_result.quality)

        return {
            "type": "partial",
            "session_id": session_id,
            "duration_seconds": round(decoding_result.duration_seconds, 2),
            "audio_quality": quality_result.quality.value,
            "gender": {
                "prediction": inference_result.gender_prediction,
                "confidence": round(inference_result.gender_confidence * confidence_multiplier, 2)
            },
            "age_bracket": {
                "prediction": inference_result.age_bracket,
                "confidence": round(inference_result.age_confidence * confidence_multiplier, 2)
            }
        }

    except Exception as e:
        logger.error(f"[{session_id}] Streaming processing error: {e}")
        return {
            "type": "error",
            "session_id": session_id,
            "error": str(e)
        }


@app.websocket("/stream")
async def stream_audio(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming with progressive predictions.

    Protocol:
    1. Client connects to ws://host:port/stream
    2. Client sends audio chunks as binary messages (WAV, MP3, or raw PCM)
    3. Server accumulates chunks and sends predictions after each update
    4. Client sends text message "done" to get final result and close

    Message types from server:
    - {"type": "connected", "session_id": "..."} - Connection established
    - {"type": "waiting", ...} - Need more audio before prediction
    - {"type": "partial", ...} - Intermediate prediction
    - {"type": "final", ...} - Final prediction (after "done")
    - {"type": "error", ...} - Error occurred

    Example client (Python):
    ```
    import websockets
    import asyncio

    async def stream_audio():
        async with websockets.connect("ws://localhost:8000/stream") as ws:
            # Send audio chunks
            with open("audio.wav", "rb") as f:
                while chunk := f.read(16000):  # 1 second chunks
                    await ws.send(chunk)
                    response = await ws.recv()
                    print(response)

            # Signal done
            await ws.send("done")
            final = await ws.recv()
            print("Final:", final)
    ```
    """
    await websocket.accept()

    session_id = str(uuid.uuid4())
    audio_buffer = bytearray()
    last_prediction_size = 0
    chunk_count = 0

    logger.info(f"[{session_id}] WebSocket streaming session started")

    # Send connection confirmation
    await websocket.send_json({
        "type": "connected",
        "session_id": session_id,
        "message": "Send audio chunks as binary, 'done' when finished"
    })

    try:
        while True:
            # Receive message (binary audio or text command)
            message = await websocket.receive()

            # Check for disconnect
            if message["type"] == "websocket.disconnect":
                break

            # Handle text messages (commands)
            if "text" in message:
                text = message["text"].strip().lower()

                if text == "done":
                    # Final prediction
                    logger.info(f"[{session_id}] Client signaled done, {len(audio_buffer)} bytes total")

                    if len(audio_buffer) > 0:
                        result = await run_in_thread_pool(
                            _process_streaming_audio,
                            bytes(audio_buffer),
                            session_id
                        )
                        result["type"] = "final"
                        result["total_chunks"] = chunk_count
                        await websocket.send_json(result)
                    else:
                        await websocket.send_json({
                            "type": "final",
                            "session_id": session_id,
                            "error": "No audio received"
                        })
                    break

                elif text == "reset":
                    # Reset buffer for new stream
                    audio_buffer = bytearray()
                    last_prediction_size = 0
                    chunk_count = 0
                    await websocket.send_json({
                        "type": "reset",
                        "session_id": session_id,
                        "message": "Buffer cleared"
                    })
                    continue

            # Handle binary messages (audio data)
            if "bytes" in message:
                chunk = message["bytes"]
                audio_buffer.extend(chunk)
                chunk_count += 1

                logger.info(f"[{session_id}] Received chunk {chunk_count}: {len(chunk)} bytes, total: {len(audio_buffer)} bytes")

                # Only process if we have minimum data (avoid decoder issues with tiny files)
                MIN_BYTES_FOR_DECODE = 48000  # ~3 seconds of 16kHz audio

                if len(audio_buffer) < MIN_BYTES_FOR_DECODE:
                    # Not enough data yet, just acknowledge
                    await websocket.send_json({
                        "type": "buffering",
                        "session_id": session_id,
                        "chunks_received": chunk_count,
                        "bytes_received": len(audio_buffer),
                        "bytes_needed": MIN_BYTES_FOR_DECODE,
                        "message": f"Buffering... ({len(audio_buffer)}/{MIN_BYTES_FOR_DECODE} bytes)"
                    })
                    continue

                # Check if we have enough new audio for an update
                bytes_per_second = 32000  # Conservative estimate
                new_bytes = len(audio_buffer) - last_prediction_size
                new_seconds = new_bytes / bytes_per_second

                if new_seconds >= UPDATE_INTERVAL_SECONDS or last_prediction_size == 0:
                    logger.info(f"[{session_id}] Processing {len(audio_buffer)} bytes...")

                    try:
                        # Run prediction with timeout
                        result = await asyncio.wait_for(
                            run_in_thread_pool(
                                _process_streaming_audio,
                                bytes(audio_buffer),
                                session_id
                            ),
                            timeout=10.0  # 10 second timeout
                        )
                        result["chunks_received"] = chunk_count
                        result["bytes_received"] = len(audio_buffer)

                        logger.info(f"[{session_id}] Prediction complete: {result.get('type')}")
                        await websocket.send_json(result)

                        # Only update last_prediction_size if we got a real prediction
                        if result.get("type") == "partial":
                            last_prediction_size = len(audio_buffer)

                    except asyncio.TimeoutError:
                        logger.error(f"[{session_id}] Processing timeout!")
                        await websocket.send_json({
                            "type": "error",
                            "session_id": session_id,
                            "error": "Processing timeout - audio may be corrupted"
                        })
                else:
                    # Not enough new audio, just acknowledge
                    await websocket.send_json({
                        "type": "buffering",
                        "session_id": session_id,
                        "chunks_received": chunk_count,
                        "bytes_received": len(audio_buffer),
                        "message": "Buffering more audio..."
                    })

    except WebSocketDisconnect:
        logger.info(f"[{session_id}] WebSocket disconnected")

    except Exception as e:
        logger.error(f"[{session_id}] WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "session_id": session_id,
                "error": str(e)
            })
        except:
            pass

    finally:
        logger.info(f"[{session_id}] WebSocket session ended, processed {chunk_count} chunks")
