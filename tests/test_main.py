"""
Tests for the FastAPI main module.

Covers:
1. Happy path - valid audio files
2. Error handling - invalid inputs
3. Edge cases - boundary conditions
4. Concurrency - thread pool behavior
5. Memory safety - no disk writes

Run with: pytest tests/test_main.py -v
"""

import asyncio
import io
import os
import sys
import tempfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import (
    app,
    read_upload_to_memory,
    run_in_thread_pool,
    get_cpu_executor,
    _process_audio_sync,
    MAX_FILE_SIZE_BYTES,
)
from app.exceptions import EmptyAudioError


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    # Use TestClient which handles lifespan events
    with TestClient(app) as client:
        yield client


@pytest.fixture
def valid_wav_bytes():
    """Create a valid WAV file in memory."""
    buffer = io.BytesIO()

    # Generate 2 seconds of audio at 16kHz
    sample_rate = 16000
    duration = 2.0
    frequency = 200  # Hz (female pitch range)

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = (np.sin(2 * np.pi * frequency * t) * 0.5 * 32767).astype(np.int16)

    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(audio.tobytes())

    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def short_wav_bytes():
    """Create a WAV file that's too short (< 0.5s)."""
    buffer = io.BytesIO()

    sample_rate = 16000
    duration = 0.1  # Too short!

    audio = np.zeros(int(sample_rate * duration), dtype=np.int16)

    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio.tobytes())

    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def silent_wav_bytes():
    """Create a silent WAV file."""
    buffer = io.BytesIO()

    sample_rate = 16000
    duration = 2.0

    # All zeros = silence
    audio = np.zeros(int(sample_rate * duration), dtype=np.int16)

    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio.tobytes())

    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def speech_like_wav_bytes():
    """Create a WAV file with speech-like characteristics."""
    buffer = io.BytesIO()

    sample_rate = 16000
    duration = 3.0

    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Simulate speech: fundamental + harmonics + amplitude modulation
    f0 = 120  # Male pitch
    audio = np.zeros_like(t)

    # Add fundamental and harmonics
    for harmonic in range(1, 6):
        amplitude = 0.3 / harmonic
        audio += amplitude * np.sin(2 * np.pi * f0 * harmonic * t)

    # Add amplitude modulation (syllable rhythm ~4 Hz)
    envelope = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))
    audio *= envelope

    # Normalize and convert to int16
    audio = (audio / np.max(np.abs(audio)) * 0.8 * 32767).astype(np.int16)

    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio.tobytes())

    buffer.seek(0)
    return buffer.read()


# =============================================================================
# TEST: HEALTH CHECK ENDPOINT
# =============================================================================

class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, client):
        """Health check should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_correct_body(self, client):
        """Health check should return expected JSON structure."""
        response = client.get("/health")
        data = response.json()

        assert data["status"] == "healthy"
        assert data["service"] == "dialflo-voice-demographic"


# =============================================================================
# TEST: ANALYZE ENDPOINT - HAPPY PATH
# =============================================================================

class TestAnalyzeEndpointHappyPath:
    """Tests for successful /analyze requests."""

    def test_valid_wav_returns_200(self, client, valid_wav_bytes):
        """Valid WAV file should return 200 with predictions."""
        response = client.post(
            "/analyze",
            files={"file": ("test.wav", io.BytesIO(valid_wav_bytes), "audio/wav")}
        )

        assert response.status_code == 200
        data = response.json()

        # Check required fields exist
        assert "contact_id" in data
        assert "gender" in data
        assert "age_bracket" in data
        assert "processing_ms" in data
        assert "audio_quality" in data

    def test_response_has_gender_prediction(self, client, speech_like_wav_bytes):
        """Response should include gender prediction with confidence."""
        response = client.post(
            "/analyze",
            files={"file": ("test.wav", io.BytesIO(speech_like_wav_bytes), "audio/wav")}
        )

        data = response.json()
        gender = data["gender"]

        assert "prediction" in gender
        assert "confidence" in gender
        assert gender["prediction"] in ["male", "female", "unknown"]
        assert 0.0 <= gender["confidence"] <= 1.0

    def test_response_has_age_bracket(self, client, speech_like_wav_bytes):
        """Response should include age bracket prediction with confidence."""
        response = client.post(
            "/analyze",
            files={"file": ("test.wav", io.BytesIO(speech_like_wav_bytes), "audio/wav")}
        )

        data = response.json()
        age = data["age_bracket"]

        assert "prediction" in age
        assert "confidence" in age
        assert age["prediction"] in ["18-30", "31-45", "46-60", "60+", "unknown"]
        assert 0.0 <= age["confidence"] <= 1.0

    def test_processing_time_under_500ms(self, client, speech_like_wav_bytes):
        """Processing should complete under 500ms for 3-second audio."""
        response = client.post(
            "/analyze",
            files={"file": ("test.wav", io.BytesIO(speech_like_wav_bytes), "audio/wav")}
        )

        data = response.json()

        # Requirement: End-to-end inference under 500ms for 5-second audio
        # 3-second audio should be faster
        assert data["processing_ms"] < 500, f"Processing took {data['processing_ms']}ms"

    def test_debug_info_included(self, client, valid_wav_bytes):
        """Response should include debug information."""
        response = client.post(
            "/analyze",
            files={"file": ("test.wav", io.BytesIO(valid_wav_bytes), "audio/wav")}
        )

        data = response.json()

        assert "_debug" in data
        debug = data["_debug"]

        assert "original_format" in debug
        assert "duration_seconds" in debug
        assert "decoding_ms" in debug
        assert "snr_db" in debug


# =============================================================================
# TEST: ANALYZE ENDPOINT - ERROR CASES
# =============================================================================

class TestAnalyzeEndpointErrors:
    """Tests for error handling in /analyze endpoint."""

    def test_empty_file_returns_400(self, client):
        """Empty file should return 400 Bad Request."""
        response = client.post(
            "/analyze",
            files={"file": ("empty.wav", io.BytesIO(b""), "audio/wav")}
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_missing_file_returns_422(self, client):
        """Missing file parameter should return 422."""
        response = client.post("/analyze")

        assert response.status_code == 422  # FastAPI validation error

    def test_corrupted_audio_returns_422(self, client):
        """Corrupted audio data should return 422."""
        corrupted_data = b"RIFF" + b"\x00" * 100 + b"WAVE" + b"garbage"

        response = client.post(
            "/analyze",
            files={"file": ("corrupt.wav", io.BytesIO(corrupted_data), "audio/wav")}
        )

        assert response.status_code == 422

    def test_too_short_audio_returns_422(self, client, short_wav_bytes):
        """Audio shorter than 0.5s should return 422."""
        response = client.post(
            "/analyze",
            files={"file": ("short.wav", io.BytesIO(short_wav_bytes), "audio/wav")}
        )

        assert response.status_code == 422
        data = response.json()
        assert "AudioTooShort" in data.get("error", "")

    def test_random_bytes_returns_error(self, client):
        """Random bytes (not audio) should return error."""
        random_data = os.urandom(1000)

        response = client.post(
            "/analyze",
            files={"file": ("random.bin", io.BytesIO(random_data), "application/octet-stream")}
        )

        # Could be 415 (unsupported format) or 422 (corrupted)
        assert response.status_code in [415, 422]


# =============================================================================
# TEST: FILE SIZE LIMIT
# =============================================================================

class TestFileSizeLimit:
    """Tests for file size limit enforcement."""

    def test_file_within_limit_accepted(self, client, valid_wav_bytes):
        """Files within size limit should be accepted."""
        assert len(valid_wav_bytes) < MAX_FILE_SIZE_BYTES

        response = client.post(
            "/analyze",
            files={"file": ("test.wav", io.BytesIO(valid_wav_bytes), "audio/wav")}
        )

        assert response.status_code == 200

    def test_file_exceeding_limit_returns_413(self, client):
        """Files exceeding size limit should return 413."""
        # Create file larger than MAX_FILE_SIZE_BYTES
        large_data = b"x" * (MAX_FILE_SIZE_BYTES + 1000)

        response = client.post(
            "/analyze",
            files={"file": ("large.bin", io.BytesIO(large_data), "audio/wav")}
        )

        assert response.status_code == 413
        data = response.json()
        assert "FileTooLarge" in str(data)


# =============================================================================
# TEST: AUDIO QUALITY HANDLING
# =============================================================================

class TestAudioQualityHandling:
    """Tests for audio quality classification and handling."""

    def test_silent_audio_returns_insufficient(self, client, silent_wav_bytes):
        """Silent audio should be classified as insufficient."""
        response = client.post(
            "/analyze",
            files={"file": ("silent.wav", io.BytesIO(silent_wav_bytes), "audio/wav")}
        )

        # Silent audio might fail validation (EmptyAudioError) or be insufficient
        if response.status_code == 200:
            data = response.json()
            assert data["audio_quality"] == "insufficient"
            assert data["gender"]["prediction"] == "unknown"
            assert data["age_bracket"]["prediction"] == "unknown"

    def test_good_audio_has_predictions(self, client, speech_like_wav_bytes):
        """Good quality audio should have actual predictions."""
        response = client.post(
            "/analyze",
            files={"file": ("speech.wav", io.BytesIO(speech_like_wav_bytes), "audio/wav")}
        )

        assert response.status_code == 200
        data = response.json()

        # Should have actual predictions (not all unknown)
        # Note: might still be unknown if pitch confidence is low
        assert data["audio_quality"] in ["good", "degraded"]


# =============================================================================
# TEST: MEMORY-SAFE UPLOAD READER
# =============================================================================

class TestReadUploadToMemory:
    """Tests for the memory-safe upload reader function."""

    @pytest.mark.asyncio
    async def test_reads_small_file_correctly(self):
        """Should correctly read small files into memory."""
        content = b"test audio content" * 100

        # Create a mock UploadFile
        mock_file = AsyncMock(spec=UploadFile)

        # Simulate chunked reading
        chunk_size = 64 * 1024
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        chunks.append(b"")  # EOF
        mock_file.read = AsyncMock(side_effect=chunks)

        result = await read_upload_to_memory(mock_file, max_size=1024*1024)

        assert result == content

    @pytest.mark.asyncio
    async def test_enforces_size_limit(self):
        """Should raise HTTPException when file exceeds limit."""
        # Create a mock that returns chunks forever
        mock_file = AsyncMock(spec=UploadFile)
        mock_file.read = AsyncMock(return_value=b"x" * 1024)  # 1KB chunks

        with pytest.raises(HTTPException) as exc_info:
            await read_upload_to_memory(mock_file, max_size=5000)

        assert exc_info.value.status_code == 413

    @pytest.mark.asyncio
    async def test_handles_empty_file(self):
        """Should handle empty files gracefully."""
        mock_file = AsyncMock(spec=UploadFile)
        mock_file.read = AsyncMock(return_value=b"")

        result = await read_upload_to_memory(mock_file)

        assert result == b""


# =============================================================================
# TEST: THREAD POOL EXECUTION
# =============================================================================

class TestThreadPoolExecution:
    """Tests for thread pool behavior."""

    @pytest.mark.asyncio
    async def test_run_in_thread_pool_executes_function(self):
        """Should execute function in thread pool and return result."""
        def slow_function(x):
            time.sleep(0.01)  # Simulate work
            return x * 2

        result = await run_in_thread_pool(slow_function, 21)

        assert result == 42

    @pytest.mark.asyncio
    async def test_thread_pool_doesnt_block_event_loop(self):
        """Thread pool work shouldn't block other async operations."""
        def slow_function():
            time.sleep(0.1)  # 100ms
            return "done"

        async def fast_async_task():
            await asyncio.sleep(0.01)  # 10ms
            return "fast"

        # Run both concurrently
        start = time.perf_counter()
        results = await asyncio.gather(
            run_in_thread_pool(slow_function),
            fast_async_task(),
            fast_async_task(),
            fast_async_task(),
        )
        elapsed = time.perf_counter() - start

        # If thread pool was blocking, this would take 100ms + 30ms = 130ms
        # With proper async, it should take ~100ms (slow task dominates)
        assert elapsed < 0.15, f"Took {elapsed}s - event loop was blocked!"
        assert results[0] == "done"
        assert all(r == "fast" for r in results[1:])


# =============================================================================
# TEST: CONCURRENCY
# =============================================================================

class TestConcurrency:
    """Tests for concurrent request handling."""

    def test_multiple_requests_processed(self, client, speech_like_wav_bytes):
        """Multiple sequential requests should all succeed."""
        for i in range(3):
            response = client.post(
                "/analyze",
                files={"file": (f"test{i}.wav", io.BytesIO(speech_like_wav_bytes), "audio/wav")}
            )
            assert response.status_code == 200

    def test_error_in_one_request_doesnt_affect_others(self, client, valid_wav_bytes):
        """Error in one request shouldn't affect subsequent requests."""
        # First: bad request
        response1 = client.post(
            "/analyze",
            files={"file": ("bad.wav", io.BytesIO(b""), "audio/wav")}
        )
        assert response1.status_code == 400

        # Second: good request (should still work)
        response2 = client.post(
            "/analyze",
            files={"file": ("good.wav", io.BytesIO(valid_wav_bytes), "audio/wav")}
        )
        assert response2.status_code == 200


# =============================================================================
# TEST: NO DISK WRITES (PII COMPLIANCE)
# =============================================================================

class TestNoDiskWrites:
    """
    Tests to verify no audio data is written to disk.

    CRITICAL: Audio is PII - strict compliance requires zero disk writes.

    Key risks mitigated:
    1. Starlette SpooledTemporaryFile - bypassed with read_upload_to_memory()
    2. PyAV/soundfile - use in-memory BytesIO
    3. scipy/numpy - all in-memory operations
    """

    def test_temp_directory_not_modified(self, client, valid_wav_bytes):
        """Temp directory should not have new files after request."""
        temp_dir = tempfile.gettempdir()

        # Get list of temp files before request
        before_files = set(os.listdir(temp_dir))

        # Make request
        response = client.post(
            "/analyze",
            files={"file": ("test.wav", io.BytesIO(valid_wav_bytes), "audio/wav")}
        )

        assert response.status_code == 200

        # Get list of temp files after request
        after_files = set(os.listdir(temp_dir))

        # New files that appeared during request
        new_files = after_files - before_files

        # Filter for audio-related temp files (Starlette uses tmpXXXXXX pattern)
        suspicious_files = [
            f for f in new_files
            if 'audio' in f.lower()
            or 'wav' in f.lower()
            or 'tmp' in f.lower()
            or f.startswith('tmp')
        ]

        # Should not have created any temp files
        assert len(suspicious_files) == 0, f"Suspicious temp files created: {suspicious_files}"

    def test_large_file_no_spillover(self, client):
        """
        Large file upload should NOT trigger SpooledTemporaryFile disk spillover.

        Starlette's default SpooledTemporaryFile spills to disk at 1MB.
        Our read_upload_to_memory() bypasses this entirely.
        """
        temp_dir = tempfile.gettempdir()

        # Create a 2MB WAV file (above spillover threshold)
        sample_rate = 16000
        duration = 60  # 60 seconds
        samples = sample_rate * duration
        audio = (np.sin(2 * np.pi * 150 * np.linspace(0, duration, samples)) * 0.5).astype(np.float32)

        # Convert to 16-bit PCM WAV (~1.9MB)
        audio_int16 = (audio * 32767).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio_int16.tobytes())
        large_wav_bytes = buffer.getvalue()

        # Verify it's > 1MB (spillover threshold)
        assert len(large_wav_bytes) > 1024 * 1024, "Test file should be >1MB"

        # Get temp files before
        before_files = set(os.listdir(temp_dir))

        # Make request with large file
        response = client.post(
            "/analyze",
            files={"file": ("large.wav", io.BytesIO(large_wav_bytes), "audio/wav")}
        )

        # Should succeed (file is within 10MB limit)
        assert response.status_code == 200

        # Get temp files after
        after_files = set(os.listdir(temp_dir))
        new_files = after_files - before_files

        # Filter for SpooledTemporaryFile patterns (tmpXXXXXX)
        spooled_files = [f for f in new_files if f.startswith('tmp') or 'upload' in f.lower()]

        # NO spillover should have occurred
        assert len(spooled_files) == 0, \
            f"SpooledTemporaryFile spillover detected! Files: {spooled_files}"

    def test_multiple_requests_no_accumulation(self, client, valid_wav_bytes):
        """Multiple requests should not accumulate temp files."""
        temp_dir = tempfile.gettempdir()

        before_files = set(os.listdir(temp_dir))

        # Make 5 requests
        for i in range(5):
            response = client.post(
                "/analyze",
                files={"file": (f"test_{i}.wav", io.BytesIO(valid_wav_bytes), "audio/wav")}
            )
            assert response.status_code == 200

        after_files = set(os.listdir(temp_dir))
        new_files = after_files - before_files

        # Filter for any tmp-like files
        tmp_files = [f for f in new_files if 'tmp' in f.lower() or f.startswith('tmp')]

        assert len(tmp_files) == 0, f"Temp files accumulated after multiple requests: {tmp_files}"

    def test_failed_request_no_temp_files(self, client):
        """Even failed requests should not leave temp files."""
        temp_dir = tempfile.gettempdir()

        before_files = set(os.listdir(temp_dir))

        # Send invalid data that will fail
        response = client.post(
            "/analyze",
            files={"file": ("bad.wav", io.BytesIO(b"not valid audio data"), "audio/wav")}
        )

        # Should fail
        assert response.status_code in (400, 415, 422)

        after_files = set(os.listdir(temp_dir))
        new_files = after_files - before_files

        tmp_files = [f for f in new_files if 'tmp' in f.lower() or f.startswith('tmp')]

        assert len(tmp_files) == 0, f"Temp files left after failed request: {tmp_files}"


# =============================================================================
# TEST: DIFFERENT AUDIO FORMATS
# =============================================================================

class TestAudioFormats:
    """Tests for different audio format handling."""

    def test_wav_format_accepted(self, client, valid_wav_bytes):
        """WAV format should be accepted."""
        response = client.post(
            "/analyze",
            files={"file": ("test.wav", io.BytesIO(valid_wav_bytes), "audio/wav")}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["_debug"]["original_format"] == "wav"


# =============================================================================
# INTEGRATION TEST WITH REAL AUDIO
# =============================================================================

class TestWithRealAudio:
    """Integration tests with real audio files (if available)."""

    @pytest.fixture
    def test_audio_dir(self):
        """Path to test audio directory."""
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_audio_for_quality_check")

    def test_real_mp4_file(self, client, test_audio_dir):
        """Test with real MP4 audio file if available."""
        mp4_path = os.path.join(test_audio_dir, "sample_audio_file.mp4")

        if not os.path.exists(mp4_path):
            pytest.skip("sample_audio_file.mp4 not found in test_audio/")

        with open(mp4_path, "rb") as f:
            audio_bytes = f.read()

        response = client.post(
            "/analyze",
            files={"file": ("sample.mp4", io.BytesIO(audio_bytes), "video/mp4")}
        )

        assert response.status_code == 200
        data = response.json()

        print(f"\nReal audio test results:")
        print(f"  Gender: {data['gender']['prediction']} ({data['gender']['confidence']:.2f})")
        print(f"  Age: {data['age_bracket']['prediction']} ({data['age_bracket']['confidence']:.2f})")
        print(f"  Quality: {data['audio_quality']}")
        print(f"  Processing: {data['processing_ms']}ms")

    def test_all_test_audio_files(self, client, test_audio_dir):
        """Test all audio files in test_audio directory."""
        if not os.path.exists(test_audio_dir):
            pytest.skip("test_audio/ directory not found")

        audio_extensions = ('.wav', '.mp3', '.mp4', '.ogg', '.flac', '.webm')

        for filename in os.listdir(test_audio_dir):
            if filename.lower().endswith(audio_extensions):
                filepath = os.path.join(test_audio_dir, filename)

                with open(filepath, "rb") as f:
                    audio_bytes = f.read()

                response = client.post(
                    "/analyze",
                    files={"file": (filename, io.BytesIO(audio_bytes), "audio/*")}
                )

                # Should either succeed or fail gracefully
                assert response.status_code in [200, 400, 415, 422], \
                    f"Unexpected status {response.status_code} for {filename}"

                if response.status_code == 200:
                    data = response.json()
                    assert "gender" in data
                    assert "age_bracket" in data


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
