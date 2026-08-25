# Dialflo Audio Inference Service

A real-time backend service that infers caller **gender** and **age bracket** from voice audio, specifically engineered for noisy logistics environments (e.g., truck cabs, warehouses, and road conditions).

## Highlights

* **Fast:** ~150-200ms end-to-end latency (SLA target: <500ms).
* **Noise-Tolerant:** 80Hz high-pass filter removes mechanical truck rumble while preserving human voice fundamentals.
* **Privacy-First:** Zero disk writes—all audio is ingested and processed entirely in-memory for strict PII compliance.
* **Quality-Aware:** Acts as an acoustic gatekeeper, automatically flagging degraded audio and scaling ML confidence scores accordingly.
* **Production-Ready:** FastAPI async architecture with ThreadPool offloading, Docker containerization, and graceful error handling.


## Quick Start

```bash
# Clone the repository
git clone https://github.com/priyadharmesh/Dialflo_Audio_Inference_Engine.git
cd Dialflo_Audio_Inference_Engine

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
  -F "file=@tts_voice_samples/01_male_young_american.mp3"
```

### Expected Response

```json
{
  "contact_id": "uuid",
  "gender": {"prediction": "male", "confidence": 0.85},
  "age_bracket": {"prediction": "31-45", "confidence": 0.72},
  "processing_ms": 156,
  "audio_quality": "good"
}
```

## API Reference

### POST /analyze

Upload an audio file for demographic analysis.

**Request:**
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@audio.wav"
```

**Success Response (200):**
```json
{
  "contact_id": "550e8400-e29b-41d4-a716-446655440000",
  "gender": {
    "prediction": "male" | "female" | "unknown",
    "confidence": 0.85
  },
  "age_bracket": {
    "prediction": "18-30" | "31-45" | "46-60" | "60+" | "unknown",
    "confidence": 0.72
  },
  "processing_ms": 156,
  "audio_quality": "good" | "degraded" | "insufficient"
}
```

**Error Response (422 - Invalid Audio):**
```json
{
  "error": "AudioTooShortError",
  "message": "Audio is 0.3s, minimum is 0.5s",
  "details": {
    "duration_seconds": 0.3,
    "minimum_seconds": 0.5
  }
}
```

**Supported Formats:** WAV, MP3, OGG, FLAC, WebM, AAC, MP4

**File Size Limit:** 10 MB

**Audio Quality Levels:**
| Level | Meaning |
|-------|---------|
| `good` | Clear audio, full confidence |
| `degraded` | Noisy but usable, reduced confidence |
| `insufficient` | Too noisy/silent, returns "unknown" |

---

### GET /health

Health check endpoint for container orchestration.

```json
{"status": "healthy", "service": "dialflo-audio-inference"}
```

## Architecture

```
Audio File
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Server                           │
│  • Async HTTP handling                                       │
│  • ThreadPoolExecutor for CPU-bound work                    │
│  • Memory-safe file upload (no disk writes)                 │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    Audio Decoder                             │
│  • Format detection via magic bytes                          │
│  • PyAV for MP3/AAC/WebM (FFmpeg bindings)                  │
│  • soundfile for WAV/FLAC/OGG                               │
│  • Output: 16kHz mono float32                               │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                   Quality Analyzer                           │
│  • 80Hz high-pass filter (removes truck rumble)             │
│  • RMS energy measurement                                    │
│  • SNR estimation                                            │
│  • Speech detection (VAD)                                    │
│  • Clipping detection                                        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                   Inference Engine                           │
│  • Pitch (F0) extraction → Gender prediction                │
│  • Speaking rate → Age estimation                           │
│  • Spectral centroid → Age estimation                       │
│  • Jitter analysis → Age estimation                         │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
JSON Response
```

## Design Decisions

### 1. Zero Disk Writes (PII Compliance)

Audio is PII. The service guarantees zero disk writes:

- Custom `read_upload_to_memory()` bypasses Starlette's `SpooledTemporaryFile` which spills to disk at 1MB
- PyAV and soundfile operate on in-memory `BytesIO` buffers
- Docker container runs with `read_only: true` filesystem

### 2. Event Loop Protection

CPU-bound audio processing blocks the async event loop. Solution:

```python
result = await loop.run_in_executor(cpu_executor, process_audio, data)
```

All heavy work runs in a `ThreadPoolExecutor`, keeping the API responsive.

### 3. 80Hz High-Pass Filter

Logistics environments have truck engine rumble (30-80Hz). We filter this while preserving speech:

| Sound Source | Frequency Range |
|--------------|-----------------|
| Truck engine | 30-80 Hz |
| Male voice | 85-180 Hz |
| Female voice | 165-255 Hz |

80Hz cutoff removes engine noise while preserving all human voice fundamentals.

### 4. Quality-Based Confidence Gating

| Quality | Action | Confidence |
|---------|--------|------------|
| GOOD | Full inference | 100% |
| DEGRADED | Inference with reduced confidence | 70% |
| INSUFFICIENT | Skip inference, return "unknown" | 0% |

This prevents unreliable predictions on poor quality audio.

### 5. Acoustic Feature-Based Inference

Uses traditional signal processing instead of deep learning:

| Approach | Pros | Cons |
|----------|------|------|
| Acoustic features | Fast, no model downloads, simple deployment | Less accurate than deep learning |
| Deep learning | More accurate | Large model (~1GB), slower, complex setup |

Trade-off: Chose simplicity and speed for this implementation.


## Design Write-Up

### Audio Decoder
**Approach:** Dual-decoder strategy with in-memory processing.

- **soundfile** for WAV/FLAC/OGG (~10-20ms) - fast native decoding
- **PyAV** for MP3/AAC/WebM (~30-50ms) - FFmpeg bindings without subprocess overhead
- **Magic byte detection** - identifies format from file content, not extension
- **BytesIO buffers** - all decoding in-memory, zero disk writes for PII compliance

### Quality Analyzer
**Approach:** Multi-metric classification with logistics-optimized filtering.

- **80Hz high-pass filter** - removes truck rumble (30-80Hz) while preserving male voice (85Hz+). Standard 200Hz would damage male speech.
- **Percentile-based VAD** - adapts to varying noise floors instead of fixed thresholds
- **Single-pass filtering** - filter once, reuse for all metrics (avoids redundant processing)
- **Three-tier output** - good/degraded/insufficient allows graceful handling rather than binary pass/fail

### Inference Engine
**Approach:** Acoustic feature extraction for fast, dependency-free inference.

- **Pitch (F0)** for gender - well-established correlation (male ~120Hz, female ~200Hz)
- **Speaking rate + spectral centroid + jitter** for age - combined features provide reasonable estimation
- **Shared computation** - pitch calculated once, reused for both gender and jitter analysis
- **Trade-off:** Chose speed and simplicity over deep learning accuracy. Architecture supports future model upgrades.

### What This Design Achieves
- **~150-200ms latency** for 5-second audio (3x faster than 500ms target)
- **Zero disk writes** throughout entire pipeline
- **Concurrent request handling** via ThreadPoolExecutor
- **Graceful degradation** - poor audio returns "unknown" rather than bad predictions

## Known Limitations

1. **Age Estimation Accuracy**: Acoustic-based age prediction is inherently limited. Speaking rate varies with emotion, fatigue, and context—not just age.

2. **Ambiguous Pitch Range**: Voices in the 160-180Hz range (overlap between male/female) receive lower confidence scores and may return "unknown".

3. **Codec Effects**: Heavy audio compression (low-bitrate VoIP) can affect spectral features, potentially impacting predictions.

4. **Language Dependency**: The acoustic features were tuned for English speakers. Accuracy may vary for other languages with different prosodic patterns.

5. **Short Audio**: Audio under 2 seconds may not have enough data for reliable age estimation.


## Performance

| Metric | Target | Actual |
|--------|--------|--------|
| End-to-end latency (5s audio) | <500ms | ~150-200ms |
| Audio decoding (WAV) | - | ~10-20ms |
| Audio decoding (MP3) | - | ~30-50ms |
| Quality analysis | - | ~5-10ms |
| Inference | - | ~50-100ms |

**Cold Start:** First request after server start takes ~200ms (after warmup). Subsequent requests are faster.

**Throughput:** Single instance handles ~50-100 concurrent requests before latency degrades.


## Privacy & PII Compliance

Audio data is treated as PII. The service ensures:

- **No disk writes**: All audio processed in-memory only
- **No storage**: Audio exists only during request processing
- **No logging of content**: Only metadata (format, duration, quality) is logged
- **No external calls**: All processing happens locally
- **Immediate cleanup**: Memory released after response is sent
- **Read-only container**: Docker runs with `read_only: true` filesystem


## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_main.py -v              # API endpoint tests
pytest tests/test_audio_decoder.py -v     # Audio decoding tests
pytest tests/test_quality_analyzer.py -v  # Quality analysis tests
pytest tests/test_inference_engine.py -v  # Inference tests

# Run integration tests with real audio files
pytest tests/test_integration_real_audio.py -v -s
```

**Test Audio Files Included:**
- `test_audio_for_quality_check/` - Quality classification samples
- `tts_voice_samples/` - TTS voice samples (MP3)


## Project Structure

```
dialflo-backend/
├── app/
│   ├── main.py                 # FastAPI application & endpoints
│   ├── exceptions.py           # Custom exception classes
│   └── services/
│       ├── audio_decoder.py    # Audio format detection & decoding
│       ├── quality_analyzer.py # Audio quality assessment
│       └── inference_engine.py # Gender & age prediction
├── tests/
│   ├── test_main.py
│   ├── test_audio_decoder.py
│   ├── test_quality_analyzer.py
│   ├── test_inference_engine.py
│   └── test_integration_real_audio.py
├── test_audio_for_quality_check/   # Test audio samples
├── tts_voice_samples/              # TTS voice samples
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```


## Requirements

**System Dependencies:**
- Python 3.11+
- FFmpeg (for audio decoding)
- libsndfile (for WAV/FLAC support)

**Key Libraries:**
- `FastAPI` - Async web framework
- `PyAV` - FFmpeg bindings for audio decoding
- `soundfile` - WAV/FLAC/OGG decoding
- `numpy` / `scipy` - Signal processing
- `uvicorn` - ASGI server

**Note:** All dependencies are included in the Docker image. No manual installation needed when using Docker.


## Future Improvements

- **Deep Learning Models**: Integrate Wav2Vec2 or similar for improved accuracy
- **Language Detection**: Add language/accent classification
- **Evaluation Harness**: Automated benchmarking against public datasets
- **GPU Support**: CUDA acceleration for deep learning inference
- **Metrics Dashboard**: Prometheus/Grafana for monitoring latency and predictions


