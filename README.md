# Dialflo Voice Demographic Service

A real-time backend service that infers caller **gender** and **age bracket** from voice audio, designed for logistics environments with background noise from trucks, warehouses, and road conditions.

## Highlights

- **Fast**: ~150-200ms end-to-end latency (target: <500ms)
- **Streaming**: WebSocket endpoint for real-time progressive predictions
- **Noise-Tolerant**: 80Hz high-pass filter removes truck rumble while preserving voice
- **Privacy-First**: Zero disk writes - all audio processed in-memory only
- **Production-Ready**: Docker containerized, health checks, graceful error handling

---

## Quick Start

```bash
# Clone the repository
git clone <your-repo-url>
cd dialflo-backend

# Start with Docker (recommended)
docker compose up --build

# Or run locally
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Analyze an audio file
curl -X POST http://localhost:8000/analyze \
  -F "file=@test_audio_for_quality_check/01_good_quality.wav"
```

---

## API Reference

### POST /analyze

Upload a complete audio file for demographic analysis.

**Request:**
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@audio.wav"
```

**Response:**
```json
{
  "contact_id": "550e8400-e29b-41d4-a716-446655440000",
  "gender": {
    "prediction": "male",
    "confidence": 0.85
  },
  "age_bracket": {
    "prediction": "31-45",
    "confidence": 0.72
  },
  "processing_ms": 156,
  "audio_quality": "good"
}
```

**Supported Formats:** WAV, MP3, OGG, FLAC, WebM, AAC, MP4

**File Size Limit:** 10 MB

---

### WebSocket /stream (Bonus Feature)

Real-time streaming endpoint for progressive predictions during live calls.

**Connection:**
```
ws://localhost:8000/stream
```

**Protocol:**
1. Client connects and receives session confirmation
2. Client sends audio chunks as binary messages
3. Server responds with progressive predictions after each chunk
4. Client sends text "done" to receive final result

**Example Client (Python):**
```python
import asyncio
import websockets

async def stream_audio():
    async with websockets.connect("ws://localhost:8000/stream") as ws:
        # Receive connection confirmation
        print(await ws.recv())
        
        # Send audio in chunks
        with open("audio.wav", "rb") as f:
            while chunk := f.read(16000):
                await ws.send(chunk)
                response = await ws.recv()
                print(response)  # Progressive prediction
        
        # Get final result
        await ws.send("done")
        print(await ws.recv())

asyncio.run(stream_audio())
```

**Response Types:**
```json
// Buffering (accumulating data)
{"type": "buffering", "bytes_received": 32000, "message": "..."}

// Partial prediction (during streaming)
{"type": "partial", "gender": {"prediction": "male", "confidence": 0.72}, ...}

// Final prediction (after "done")
{"type": "final", "gender": {"prediction": "male", "confidence": 0.85}, ...}
```

---

### GET /health

Health check endpoint for container orchestration.

```json
{"status": "healthy", "service": "dialflo-voice-demographic"}
```

---

## Architecture

```
                         ┌─────────────────────────────────┐
      HTTP POST ────────▶│                                 │
                         │        FastAPI Server           │
      WebSocket ────────▶│   (ThreadPoolExecutor for CPU)  │
                         │                                 │
                         └───────────────┬─────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
           ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
           │    Audio     │    │   Quality    │    │  Inference   │
           │   Decoder    │───▶│   Analyzer   │───▶│   Engine     │
           └──────────────┘    └──────────────┘    └──────────────┘
                 │                    │                    │
                 ▼                    ▼                    ▼
           • PyAV (MP3/AAC)    • 80Hz high-pass     • Pitch (F0)
           • soundfile (WAV)   • SNR estimation     • Speaking rate
           • 16kHz mono out    • Speech detection   • Spectral centroid
                               • Clipping check     • Jitter analysis
```

---

## Design Decisions

### 1. Zero Disk Writes (PII Compliance)

Audio is PII. The service guarantees zero disk writes:

- Custom `read_upload_to_memory()` bypasses Starlette's `SpooledTemporaryFile` (which spills to disk at 1MB)
- PyAV and soundfile operate on in-memory `BytesIO` buffers
- Docker container runs with `read_only: true` filesystem
- All numpy/scipy operations are purely in-memory

### 2. Event Loop Protection

CPU-bound audio processing would block FastAPI's async event loop. All heavy work runs in a `ThreadPoolExecutor`:

```python
result = await loop.run_in_executor(cpu_executor, process_audio, data)
```

### 3. 80Hz High-Pass Filter (Not 200Hz)

Logistics environments have truck engine rumble (30-80Hz). We filter this while preserving human speech:

| Sound Source | Frequency Range |
|--------------|-----------------|
| Truck engine | 30-80 Hz |
| Male voice | 85-180 Hz |
| Female voice | 165-255 Hz |

An **80Hz cutoff** removes engine noise while preserving ALL human voice fundamentals.

### 4. Quality-Based Confidence Gating

Audio quality affects prediction reliability:

| Quality | Action | Confidence Multiplier |
|---------|--------|----------------------|
| GOOD | Full inference | 100% |
| DEGRADED | Inference with warning | 70% |
| INSUFFICIENT | Skip inference, return "unknown" | 0% |

This prevents bad predictions on unusable audio.

### 5. Acoustic Feature-Based Inference

The current implementation uses traditional acoustic features:

| Feature | Used For |
|---------|----------|
| Pitch (F0) | Gender prediction |
| Speaking rate | Age estimation |
| Spectral centroid | Age estimation |
| Jitter | Age estimation |

**Trade-offs:**
- ✅ No external model downloads (~1GB)
- ✅ Fast inference (<100ms)
- ✅ Simple deployment
- ❌ Less accurate than deep learning
- ❌ Age estimation is inherently limited

---

## Design Write-Up

**Approach:** I chose acoustic feature extraction over deep learning models for practical deployment reasons. Pitch-based gender detection is well-established (male ~120Hz, female ~200Hz), and while age estimation from acoustics is less reliable, combining speaking rate, spectral centroid, and jitter provides reasonable baseline predictions without requiring large model downloads.

**Libraries:** FastAPI provides async HTTP handling while ThreadPoolExecutor offloads CPU work. PyAV (FFmpeg bindings) handles diverse audio formats in-memory without subprocess overhead. scipy provides signal processing primitives (filtering, FFT) that are fast and well-tested.

**Improvements with more time:** I would integrate a Wav2Vec2-based ONNX model for significantly better accuracy, especially for age prediction. The current architecture already supports this - the InferenceEngine class can be swapped without changing the API layer.

**Scaling to 1,000 concurrent calls:** The current single-instance design handles ~50-100 concurrent requests. For 1,000 concurrent calls, I would: (1) run multiple container replicas behind a load balancer, (2) use Kubernetes horizontal pod autoscaling based on CPU utilization, (3) consider GPU inference if using deep learning models. The stateless design and in-memory processing make horizontal scaling straightforward.

---

## Known Limitations

1. **Age Estimation Accuracy**: Acoustic-based age prediction is inherently limited. Speaking rate varies with emotion, fatigue, and context - not just age.

2. **Ambiguous Pitch Range**: Voices in the 160-180Hz range (overlap between male/female) receive lower confidence scores and may be classified as "unknown".

3. **Codec Effects**: Heavy audio compression (low-bitrate VoIP) can affect spectral features, potentially impacting age estimation.

4. **Non-Speech Audio**: Pure noise, music, or silence correctly returns "unknown" but may briefly show as "degraded" quality during streaming.

5. **WebSocket Partial Decoding**: Container formats (MP4, M4A) may fail to decode when partially received; WAV and MP3 work better for streaming.

---

## Performance

| Metric | Target | Actual |
|--------|--------|--------|
| End-to-end latency (5s audio) | <500ms | ~150-200ms |
| Audio decoding (WAV) | - | ~10-20ms |
| Audio decoding (MP3) | - | ~30-50ms |
| Quality analysis | - | ~5-10ms |
| Inference | - | ~50-100ms |
| First request (cold start) | - | ~200ms (after warmup) |

---

## Privacy & PII Compliance

- **No audio storage**: Audio exists only in memory during request processing
- **No logging of audio content**: Only metadata (format, duration, quality) is logged
- **No external API calls**: All processing happens locally
- **Read-only container**: Docker runs with `read_only: true`
- **Automatic cleanup**: Memory released immediately after response

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_main.py -v              # API tests
pytest tests/test_audio_decoder.py -v     # Decoder tests
pytest tests/test_quality_analyzer.py -v  # Quality tests
pytest tests/test_inference_engine.py -v  # Inference tests

# Run integration tests with real audio
pytest tests/test_integration_real_audio.py -v -s

# Test WebSocket streaming
pip install websockets
python test_websocket.py test_audio_for_quality_check/01_good_quality.wav
```

**Test Coverage:**
- 100+ unit tests
- Integration tests with real audio files
- PII compliance tests (verify no disk writes)
- Performance tests (verify <500ms latency)

---

## Project Structure

```
dialflo-backend/
├── app/
│   ├── main.py                 # FastAPI app, endpoints, WebSocket
│   ├── exceptions.py           # Custom exception classes
│   └── services/
│       ├── audio_decoder.py    # Format detection, decoding
│       ├── quality_analyzer.py # Audio quality assessment
│       └── inference_engine.py # Gender/age prediction
├── tests/
│   ├── test_main.py
│   ├── test_audio_decoder.py
│   ├── test_quality_analyzer.py
│   ├── test_inference_engine.py
│   └── test_integration_real_audio.py
├── test_audio_for_quality_check/  # Quality test samples
├── comprehensive_voice_tests/      # Gender/age test samples
├── tts_voice_samples/              # TTS samples (MP3)
├── test_websocket.py              # WebSocket test client
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.11+
- FFmpeg (for PyAV audio decoding)
- libsndfile (for soundfile WAV decoding)

All dependencies are included in the Docker image.

---

## Future Improvements

1. **Deep Learning Models**: Replace acoustic heuristics with Wav2Vec2 or similar for better accuracy
2. **Language Detection**: Add best-effort language/accent classification
3. **Evaluation Harness**: Automated accuracy benchmarking against Mozilla Common Voice
4. **GPU Support**: CUDA acceleration for deep learning inference
5. **Prometheus Metrics**: Expose latency histograms and prediction distributions

---

## License

Proprietary - Dialflo Engineering Assignment
