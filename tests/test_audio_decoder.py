"""
Tests for the Audio Decoder module.

Covers:
1. Format detection (magic bytes)
2. Decoding different formats (WAV, MP3, OGG, etc.)
3. Mono conversion
4. Resampling
5. Validation (duration, silence)
6. Edge cases and error handling
7. Performance benchmarks

Run with: pytest tests/test_audio_decoder.py -v
"""

import io
import os
import sys
import time
import wave
from typing import Tuple

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.audio_decoder import (
    AudioFormat,
    DecodingResult,
    TARGET_SAMPLE_RATE,
    MIN_DURATION_SECONDS,
    MAX_DURATION_SECONDS,
    detect_format,
    decode_audio_bytes,
    decode_audio_bytes_detailed,
    _to_mono,
    _resample,
    _validate,
)
from app.exceptions import (
    AudioTooShortError,
    AudioTooLongError,
    CorruptedAudioError,
    EmptyAudioError,
    UnsupportedFormatError,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_wav_bytes(
    duration: float = 2.0,
    sample_rate: int = 16000,
    frequency: float = 200.0,
    channels: int = 1,
    silent: bool = False
) -> bytes:
    """Create a WAV file in memory and return as bytes."""
    buffer = io.BytesIO()

    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, False)

    if silent:
        audio = np.zeros(num_samples, dtype=np.float32)
    else:
        audio = (np.sin(2 * np.pi * frequency * t) * 0.5).astype(np.float32)

    # Convert to int16 for WAV
    audio_int16 = (audio * 32767).astype(np.int16)

    # Handle stereo
    if channels == 2:
        audio_int16 = np.column_stack([audio_int16, audio_int16])

    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(audio_int16.tobytes())

    buffer.seek(0)
    return buffer.read()


def create_speech_like_wav(
    duration: float = 3.0,
    sample_rate: int = 16000,
    fundamental: float = 120.0  # Male voice ~120Hz
) -> bytes:
    """Create a WAV file with speech-like characteristics."""
    buffer = io.BytesIO()

    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, False)

    # Generate harmonics (speech has multiple harmonics)
    audio = np.zeros(num_samples, dtype=np.float32)
    for harmonic in range(1, 8):
        amplitude = 0.3 / harmonic
        audio += amplitude * np.sin(2 * np.pi * fundamental * harmonic * t)

    # Add amplitude modulation (syllable rhythm ~4 Hz)
    envelope = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))
    audio *= envelope

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8

    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_int16.tobytes())

    buffer.seek(0)
    return buffer.read()


# =============================================================================
# TEST: FORMAT DETECTION
# =============================================================================

class TestFormatDetection:
    """Tests for the detect_format function."""

    def test_detect_wav_format(self):
        """Should detect WAV format from RIFF header."""
        wav_bytes = create_wav_bytes()
        assert detect_format(wav_bytes) == AudioFormat.WAV

    def test_detect_wav_magic_bytes(self):
        """WAV files start with RIFF....WAVE."""
        # Minimal WAV header
        wav_header = b'RIFF' + b'\x00\x00\x00\x00' + b'WAVE'
        assert detect_format(wav_header) == AudioFormat.WAV

    def test_detect_mp3_id3_tag(self):
        """MP3 files with ID3 tag start with 'ID3'."""
        mp3_with_id3 = b'ID3' + b'\x00' * 20
        assert detect_format(mp3_with_id3) == AudioFormat.MP3

    def test_detect_mp3_frame_sync(self):
        """MP3 frames start with 0xFF 0xE0-0xEF (sync word)."""
        # 0xFF 0xFB = valid MP3 frame sync (MPEG1 Layer3)
        mp3_frame = b'\xff\xfb' + b'\x00' * 20
        assert detect_format(mp3_frame) == AudioFormat.MP3

    def test_detect_ogg_format(self):
        """OGG files start with 'OggS'."""
        ogg_header = b'OggS' + b'\x00' * 20
        assert detect_format(ogg_header) == AudioFormat.OGG

    def test_detect_flac_format(self):
        """FLAC files start with 'fLaC'."""
        flac_header = b'fLaC' + b'\x00' * 20
        assert detect_format(flac_header) == AudioFormat.FLAC

    def test_detect_webm_format(self):
        """WebM files start with EBML header."""
        webm_header = b'\x1a\x45\xdf\xa3' + b'\x00' * 20
        assert detect_format(webm_header) == AudioFormat.WEBM

    def test_detect_adts_aac(self):
        """ADTS AAC streams start with 0xFFF sync word."""
        # ADTS header: 0xFFF sync + ID=0 + layer=00 + protection
        # 0xFF 0xF1 = sync + MPEG4 + layer 0 + no CRC
        adts_header = b'\xff\xf1' + b'\x00' * 20
        assert detect_format(adts_header) == AudioFormat.AAC

    def test_detect_mp4_container(self):
        """MP4 files have 'ftyp' atom."""
        # Standard MP4 with ftyp at byte 4
        mp4_header = b'\x00\x00\x00\x14ftypisom' + b'\x00' * 10
        result = detect_format(mp4_header)
        assert result in (AudioFormat.MP4, AudioFormat.AAC)

    def test_detect_m4a_audio(self):
        """M4A files (AAC in MP4) have 'M4A ' brand."""
        m4a_header = b'\x00\x00\x00\x14ftypM4A ' + b'\x00' * 10
        assert detect_format(m4a_header) == AudioFormat.AAC

    def test_detect_amr_format(self):
        """AMR files start with '#!AMR'."""
        amr_header = b'#!AMR\n' + b'\x00' * 20
        assert detect_format(amr_header) == AudioFormat.AMR

    def test_detect_unknown_format(self):
        """Unknown/random bytes should return UNKNOWN."""
        random_bytes = os.urandom(100)
        # Make sure it doesn't match any known format
        if random_bytes[:4] not in (b'RIFF', b'OggS', b'fLaC', b'ID3'):
            assert detect_format(random_bytes) == AudioFormat.UNKNOWN

    def test_detect_short_input(self):
        """Very short input should return UNKNOWN."""
        assert detect_format(b'') == AudioFormat.UNKNOWN
        assert detect_format(b'R') == AudioFormat.UNKNOWN
        assert detect_format(b'RIFF') == AudioFormat.UNKNOWN  # Need 12 bytes

    def test_mp3_vs_aac_differentiation(self):
        """Should differentiate MP3 (0xFF 0xE0-EF) from AAC (0xFF 0xF0-F9)."""
        # MP3: 0xFF 0xE2 (layer bits != 00)
        mp3_sync = b'\xff\xe2' + b'\x00' * 20
        assert detect_format(mp3_sync) == AudioFormat.MP3

        # AAC: 0xFF 0xF1 (layer bits == 00)
        aac_sync = b'\xff\xf1' + b'\x00' * 20
        assert detect_format(aac_sync) == AudioFormat.AAC


# =============================================================================
# TEST: MONO CONVERSION
# =============================================================================

class TestMonoConversion:
    """Tests for the _to_mono function."""

    def test_mono_unchanged(self):
        """Mono audio should pass through unchanged."""
        mono = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        result = _to_mono(mono)
        np.testing.assert_array_equal(result, mono)

    def test_stereo_samples_channels(self):
        """Stereo (samples, channels) should average to mono."""
        # Shape: (5 samples, 2 channels)
        stereo = np.array([
            [1.0, 3.0],
            [2.0, 4.0],
            [3.0, 5.0],
            [4.0, 6.0],
            [5.0, 7.0],
        ], dtype=np.float32)

        result = _to_mono(stereo)

        # Average: (1+3)/2=2, (2+4)/2=3, etc.
        expected = np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_stereo_channels_samples(self):
        """Stereo (channels, samples) should average to mono."""
        # Shape: (2 channels, 5 samples)
        stereo = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
        ], dtype=np.float32)

        result = _to_mono(stereo)

        expected = np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_5_1_surround(self):
        """5.1 surround (6 channels) should average to mono."""
        # Shape: (6 channels, 100 samples)
        surround = np.random.rand(6, 100).astype(np.float32)

        result = _to_mono(surround)

        assert result.ndim == 1
        assert len(result) == 100
        # Should be average of all 6 channels
        expected = np.mean(surround, axis=0)
        np.testing.assert_array_almost_equal(result, expected)


# =============================================================================
# TEST: RESAMPLING
# =============================================================================

class TestResampling:
    """Tests for the _resample function."""

    def test_no_resample_needed(self):
        """Same sample rate should return unchanged."""
        audio = np.random.rand(16000).astype(np.float32)
        result = _resample(audio, 16000, 16000)
        np.testing.assert_array_equal(result, audio)

    def test_downsample_44100_to_16000(self):
        """Should downsample from 44.1kHz to 16kHz."""
        # 1 second at 44.1kHz
        audio = np.random.rand(44100).astype(np.float32)
        result = _resample(audio, 44100, 16000)

        # Output should be ~16000 samples (1 second at 16kHz)
        # Allow small tolerance for resampling
        assert abs(len(result) - 16000) < 10

    def test_downsample_48000_to_16000(self):
        """Should downsample from 48kHz to 16kHz."""
        audio = np.random.rand(48000).astype(np.float32)
        result = _resample(audio, 48000, 16000)

        # 48000/16000 = 3, so output = 16000 exactly
        assert len(result) == 16000

    def test_upsample_8000_to_16000(self):
        """Should upsample from 8kHz to 16kHz."""
        audio = np.random.rand(8000).astype(np.float32)
        result = _resample(audio, 8000, 16000)

        # 8000 * 2 = 16000
        assert len(result) == 16000

    def test_resample_preserves_dtype(self):
        """Output should be float32."""
        audio = np.random.rand(44100).astype(np.float32)
        result = _resample(audio, 44100, 16000)
        assert result.dtype == np.float32


# =============================================================================
# TEST: VALIDATION
# =============================================================================

class TestValidation:
    """Tests for the _validate function."""

    def test_valid_audio_passes(self):
        """Valid audio should pass validation."""
        # 2 seconds at 16kHz, non-silent
        audio = np.sin(np.linspace(0, 10, 32000)).astype(np.float32) * 0.5
        _validate(audio, 16000)  # Should not raise

    def test_empty_audio_raises(self):
        """Empty audio should raise EmptyAudioError."""
        audio = np.array([], dtype=np.float32)
        with pytest.raises(EmptyAudioError):
            _validate(audio, 16000)

    def test_too_short_audio_raises(self):
        """Audio < 0.5s should raise AudioTooShortError."""
        # 0.1 seconds
        audio = np.random.rand(1600).astype(np.float32)
        with pytest.raises(AudioTooShortError) as exc_info:
            _validate(audio, 16000)

        assert exc_info.value.details["duration_seconds"] == pytest.approx(0.1, rel=0.1)
        assert exc_info.value.details["minimum_seconds"] == MIN_DURATION_SECONDS

    def test_too_long_audio_raises(self):
        """Audio > 60s should raise AudioTooLongError."""
        # 65 seconds
        audio = np.random.rand(16000 * 65).astype(np.float32)
        with pytest.raises(AudioTooLongError) as exc_info:
            _validate(audio, 16000)

        assert exc_info.value.details["maximum_seconds"] == MAX_DURATION_SECONDS

    def test_silent_audio_raises(self):
        """Silent audio (all zeros) should raise EmptyAudioError."""
        audio = np.zeros(32000, dtype=np.float32)
        with pytest.raises(EmptyAudioError) as exc_info:
            _validate(audio, 16000)

        assert "silent" in exc_info.value.message.lower()

    def test_near_silent_audio_raises(self):
        """Very quiet audio should raise EmptyAudioError."""
        # RMS < 1e-6
        audio = np.full(32000, 1e-8, dtype=np.float32)
        with pytest.raises(EmptyAudioError):
            _validate(audio, 16000)

    def test_boundary_duration_min(self):
        """Audio exactly at minimum duration should pass."""
        # Exactly 0.5 seconds
        audio = np.random.rand(8000).astype(np.float32)
        _validate(audio, 16000)  # Should not raise

    def test_boundary_duration_max(self):
        """Audio exactly at maximum duration should pass."""
        # Exactly 60 seconds
        audio = np.random.rand(16000 * 60).astype(np.float32)
        _validate(audio, 16000)  # Should not raise


# =============================================================================
# TEST: FULL DECODING
# =============================================================================

class TestDecoding:
    """Tests for the main decode functions."""

    def test_decode_valid_wav(self):
        """Should decode valid WAV file."""
        wav_bytes = create_wav_bytes(duration=2.0)
        audio, sr = decode_audio_bytes(wav_bytes)

        assert sr == TARGET_SAMPLE_RATE
        assert audio.dtype == np.float32
        assert len(audio) == pytest.approx(32000, rel=0.01)  # 2s at 16kHz

    def test_decode_wav_different_sample_rates(self):
        """Should handle WAV files at different sample rates."""
        for orig_sr in [8000, 22050, 44100, 48000]:
            wav_bytes = create_wav_bytes(duration=1.0, sample_rate=orig_sr)
            audio, sr = decode_audio_bytes(wav_bytes)

            assert sr == TARGET_SAMPLE_RATE
            # Duration should be preserved (~1 second)
            assert len(audio) == pytest.approx(16000, rel=0.05)

    def test_decode_stereo_wav(self):
        """Should convert stereo WAV to mono."""
        wav_bytes = create_wav_bytes(duration=2.0, channels=2)
        audio, sr = decode_audio_bytes(wav_bytes)

        assert audio.ndim == 1  # Mono
        assert sr == TARGET_SAMPLE_RATE

    def test_decode_returns_detailed_result(self):
        """decode_audio_bytes_detailed should return DecodingResult."""
        wav_bytes = create_wav_bytes(duration=2.0, sample_rate=44100)
        result = decode_audio_bytes_detailed(wav_bytes)

        assert isinstance(result, DecodingResult)
        assert result.sample_rate == TARGET_SAMPLE_RATE
        assert result.original_format == AudioFormat.WAV
        assert result.original_sample_rate == 44100
        assert result.duration_seconds == pytest.approx(2.0, rel=0.05)
        assert result.decoding_ms > 0

    def test_decode_empty_bytes_raises(self):
        """Empty bytes should raise EmptyAudioError."""
        with pytest.raises(EmptyAudioError):
            decode_audio_bytes(b'')

    def test_decode_corrupted_raises(self):
        """Corrupted data should raise CorruptedAudioError."""
        # Start with valid WAV header but garbage data
        corrupted = b'RIFF' + b'\xff' * 100 + b'WAVE' + b'garbage'

        with pytest.raises((CorruptedAudioError, EmptyAudioError)):
            decode_audio_bytes(corrupted)

    def test_decode_random_bytes_raises(self):
        """Random bytes should raise appropriate error."""
        random_data = os.urandom(1000)

        with pytest.raises((UnsupportedFormatError, CorruptedAudioError)):
            decode_audio_bytes(random_data)

    def test_decode_too_short_raises(self):
        """Audio too short should raise AudioTooShortError."""
        # 0.1 second
        wav_bytes = create_wav_bytes(duration=0.1)

        with pytest.raises(AudioTooShortError):
            decode_audio_bytes(wav_bytes)

    def test_decode_silent_raises(self):
        """Silent audio should raise EmptyAudioError."""
        wav_bytes = create_wav_bytes(duration=2.0, silent=True)

        with pytest.raises(EmptyAudioError):
            decode_audio_bytes(wav_bytes)

    def test_decode_with_validation_disabled(self):
        """Should skip validation when validate=False."""
        # Too short, but validation disabled
        wav_bytes = create_wav_bytes(duration=0.1)
        audio, sr = decode_audio_bytes(wav_bytes, validate=False)

        assert len(audio) > 0  # Should succeed


# =============================================================================
# TEST: PERFORMANCE
# =============================================================================

class TestPerformance:
    """Performance benchmarks for the decoder."""

    def test_wav_decode_under_50ms(self):
        """WAV decoding should complete in under 50ms."""
        # 5 second audio
        wav_bytes = create_wav_bytes(duration=5.0)

        start = time.perf_counter()
        decode_audio_bytes(wav_bytes)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, f"WAV decode took {elapsed_ms:.1f}ms"

    def test_speech_like_wav_decode_performance(self):
        """Speech-like WAV should decode quickly."""
        wav_bytes = create_speech_like_wav(duration=5.0)

        start = time.perf_counter()
        result = decode_audio_bytes_detailed(wav_bytes)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Internal timing should be consistent
        assert result.decoding_ms < 100
        assert elapsed_ms < 100

    def test_high_sample_rate_resample_performance(self):
        """48kHz -> 16kHz resampling should be fast."""
        # 5 seconds at 48kHz = 240,000 samples
        wav_bytes = create_wav_bytes(duration=5.0, sample_rate=48000)

        start = time.perf_counter()
        decode_audio_bytes(wav_bytes)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, f"48kHz resample took {elapsed_ms:.1f}ms"


# =============================================================================
# TEST: INTEGRATION WITH REAL FILES
# =============================================================================

class TestRealFiles:
    """Integration tests with real audio files."""

    @pytest.fixture
    def test_audio_dir(self):
        """Path to test audio directory."""
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_audio_for_quality_check")

    def test_decode_real_wav(self, test_audio_dir):
        """Test decoding real WAV file."""
        wav_path = os.path.join(test_audio_dir, "01_good_quality.wav")

        if not os.path.exists(wav_path):
            pytest.skip("01_good_quality.wav not found")

        with open(wav_path, "rb") as f:
            wav_bytes = f.read()

        result = decode_audio_bytes_detailed(wav_bytes)

        assert result.original_format == AudioFormat.WAV
        assert result.sample_rate == TARGET_SAMPLE_RATE
        assert result.duration_seconds > 0

        print(f"\nReal WAV: {result.duration_seconds:.2f}s, "
              f"orig_sr={result.original_sample_rate}, "
              f"decoded in {result.decoding_ms:.1f}ms")

    def test_decode_real_mp4(self, test_audio_dir):
        """Test decoding real MP4 file."""
        mp4_path = os.path.join(test_audio_dir, "sample_audio_file.mp4")

        if not os.path.exists(mp4_path):
            pytest.skip("sample_audio_file.mp4 not found")

        with open(mp4_path, "rb") as f:
            mp4_bytes = f.read()

        result = decode_audio_bytes_detailed(mp4_bytes)

        assert result.original_format in (AudioFormat.MP4, AudioFormat.AAC)
        assert result.sample_rate == TARGET_SAMPLE_RATE

        print(f"\nReal MP4: {result.duration_seconds:.2f}s, "
              f"orig_sr={result.original_sample_rate}, "
              f"decoded in {result.decoding_ms:.1f}ms")

    def test_decode_all_test_files(self, test_audio_dir):
        """Test decoding all audio files in test directory."""
        if not os.path.exists(test_audio_dir):
            pytest.skip("test_audio/ directory not found")

        audio_extensions = ('.wav', '.mp3', '.mp4', '.ogg', '.flac', '.webm', '.m4a')
        results = []

        for filename in sorted(os.listdir(test_audio_dir)):
            if filename.lower().endswith(audio_extensions):
                filepath = os.path.join(test_audio_dir, filename)

                with open(filepath, "rb") as f:
                    audio_bytes = f.read()

                try:
                    result = decode_audio_bytes_detailed(audio_bytes, validate=False)
                    results.append((filename, "OK", result))
                except Exception as e:
                    results.append((filename, "FAIL", str(e)))

        # Print summary
        print(f"\n{'File':<40} {'Status':<10} {'Format':<10} {'Duration':<10} {'Time':<10}")
        print("-" * 80)

        for filename, status, data in results:
            if status == "OK":
                print(f"{filename:<40} {status:<10} {data.original_format.value:<10} "
                      f"{data.duration_seconds:<10.2f} {data.decoding_ms:<10.1f}ms")
            else:
                print(f"{filename:<40} {status:<10} {data[:30]}")


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
