"""
Integration Tests with Real Audio Files

Tests the complete pipeline (decode → quality → inference) with actual audio samples.

Test folders:
1. test_audio_for_quality_check/ - Quality classification samples
2. comprehensive_voice_tests/ - Gender and age prediction samples
3. tts_voice_samples/ - TTS samples in MP3 format (accent variations)

Run with: pytest tests/test_integration_real_audio.py -v
"""

import io
import os
import sys
import time
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from app.main import app
from app.services.audio_decoder import decode_audio_bytes_detailed
from app.services.quality_analyzer import analyze_quality, AudioQuality
from app.services.inference_engine import init_engine, get_engine, InferenceEngine


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def client():
    """Create test client."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def engine():
    """Initialize inference engine."""
    return init_engine()


# =============================================================================
# PATH HELPERS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent

QUALITY_TEST_DIR = PROJECT_ROOT / "test_audio_for_quality_check"
VOICE_TEST_DIR = PROJECT_ROOT / "comprehensive_voice_tests"
TTS_TEST_DIR = PROJECT_ROOT / "tts_voice_samples"


def get_audio_files(directory: Path, extensions=(".wav", ".mp3")):
    """Get all audio files in a directory."""
    if not directory.exists():
        return []
    files = []
    for ext in extensions:
        files.extend(directory.glob(f"*{ext}"))
    return sorted(files)


# =============================================================================
# TEST: QUALITY CLASSIFICATION WITH REAL AUDIO
# =============================================================================

class TestQualityWithRealAudio:
    """Test quality analyzer with real audio samples."""

    @pytest.fixture(autouse=True)
    def check_test_files(self):
        """Skip if test files don't exist."""
        if not QUALITY_TEST_DIR.exists():
            pytest.skip(f"Quality test directory not found: {QUALITY_TEST_DIR}")

    def test_good_quality_sample(self):
        """01_good_quality.wav should be classified as GOOD."""
        audio_path = QUALITY_TEST_DIR / "01_good_quality.wav"
        if not audio_path.exists():
            pytest.skip(f"File not found: {audio_path}")

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        result = decode_audio_bytes_detailed(audio_bytes)
        quality = analyze_quality(result.audio, result.sample_rate)

        print(f"\n  File: {audio_path.name}")
        print(f"  Quality: {quality.quality.value}")
        print(f"  RMS: {quality.rms_energy:.4f}, SNR: {quality.snr_estimate_db:.1f}dB")
        print(f"  Speech ratio: {quality.speech_ratio:.2f}")

        assert quality.quality == AudioQuality.GOOD, \
            f"Expected GOOD, got {quality.quality.value}"

    def test_degraded_noisy_sample(self):
        """02_degraded_noisy.wav should be DEGRADED."""
        audio_path = QUALITY_TEST_DIR / "02_degraded_noisy.wav"
        if not audio_path.exists():
            pytest.skip(f"File not found: {audio_path}")

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        result = decode_audio_bytes_detailed(audio_bytes)
        quality = analyze_quality(result.audio, result.sample_rate)

        print(f"\n  File: {audio_path.name}")
        print(f"  Quality: {quality.quality.value}")
        print(f"  SNR: {quality.snr_estimate_db:.1f}dB")

        assert quality.quality in (AudioQuality.DEGRADED, AudioQuality.INSUFFICIENT), \
            f"Expected DEGRADED/INSUFFICIENT, got {quality.quality.value}"

    def test_degraded_quiet_sample(self):
        """03_degraded_quiet.wav should be DEGRADED."""
        audio_path = QUALITY_TEST_DIR / "03_degraded_quiet.wav"
        if not audio_path.exists():
            pytest.skip(f"File not found: {audio_path}")

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        result = decode_audio_bytes_detailed(audio_bytes)
        quality = analyze_quality(result.audio, result.sample_rate)

        print(f"\n  File: {audio_path.name}")
        print(f"  Quality: {quality.quality.value}")
        print(f"  RMS: {quality.rms_energy:.4f}")

        assert quality.quality in (AudioQuality.DEGRADED, AudioQuality.INSUFFICIENT), \
            f"Expected DEGRADED/INSUFFICIENT, got {quality.quality.value}"

    def test_insufficient_noise_only(self):
        """04_insufficient_noise_only.wav should be INSUFFICIENT."""
        audio_path = QUALITY_TEST_DIR / "04_insufficient_noise_only.wav"
        if not audio_path.exists():
            pytest.skip(f"File not found: {audio_path}")

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        result = decode_audio_bytes_detailed(audio_bytes)
        quality = analyze_quality(result.audio, result.sample_rate)

        print(f"\n  File: {audio_path.name}")
        print(f"  Quality: {quality.quality.value}")

        # Noise-only should be INSUFFICIENT or DEGRADED
        assert quality.quality in (AudioQuality.INSUFFICIENT, AudioQuality.DEGRADED), \
            f"Expected INSUFFICIENT/DEGRADED, got {quality.quality.value}"

    def test_insufficient_silence(self):
        """05_insufficient_silence.wav should be INSUFFICIENT."""
        audio_path = QUALITY_TEST_DIR / "05_insufficient_silence.wav"
        if not audio_path.exists():
            pytest.skip(f"File not found: {audio_path}")

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        result = decode_audio_bytes_detailed(audio_bytes)
        quality = analyze_quality(result.audio, result.sample_rate)

        print(f"\n  File: {audio_path.name}")
        print(f"  Quality: {quality.quality.value}")
        print(f"  RMS: {quality.rms_energy:.6f}")

        assert quality.quality == AudioQuality.INSUFFICIENT, \
            f"Silence should be INSUFFICIENT, got {quality.quality.value}"

    def test_insufficient_clipped(self):
        """06_insufficient_clipped.wav should be INSUFFICIENT or DEGRADED."""
        audio_path = QUALITY_TEST_DIR / "06_insufficient_clipped.wav"
        if not audio_path.exists():
            pytest.skip(f"File not found: {audio_path}")

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        result = decode_audio_bytes_detailed(audio_bytes)
        quality = analyze_quality(result.audio, result.sample_rate)

        print(f"\n  File: {audio_path.name}")
        print(f"  Quality: {quality.quality.value}")
        print(f"  Clipping ratio: {quality.clipping_ratio:.4f}")

        # Heavily clipped should be INSUFFICIENT or DEGRADED
        assert quality.quality in (AudioQuality.INSUFFICIENT, AudioQuality.DEGRADED), \
            f"Clipped audio should be INSUFFICIENT/DEGRADED, got {quality.quality.value}"


# =============================================================================
# TEST: GENDER PREDICTION WITH REAL AUDIO
# =============================================================================

class TestGenderWithRealAudio:
    """Test gender prediction with real voice samples."""

    @pytest.fixture(autouse=True)
    def check_test_files(self):
        """Skip if test files don't exist."""
        if not VOICE_TEST_DIR.exists() and not TTS_TEST_DIR.exists():
            pytest.skip("Voice test directories not found")

    def _predict_file(self, audio_path: Path, engine: InferenceEngine):
        """Helper to predict gender from audio file."""
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        result = decode_audio_bytes_detailed(audio_bytes)
        prediction = engine.predict(result.audio, result.sample_rate)
        return prediction

    def test_male_voices_detected(self, engine):
        """Male voice samples should be predicted as male."""
        male_files = [
            VOICE_TEST_DIR / "01_male_very_young.wav",
            VOICE_TEST_DIR / "02_male_young_adult.wav",
            VOICE_TEST_DIR / "03_male_middle_aged.wav",
            VOICE_TEST_DIR / "04_male_older.wav",
            VOICE_TEST_DIR / "15_male_very_deep.wav",
            TTS_TEST_DIR / "01_male_young_american.mp3",
            TTS_TEST_DIR / "07_male_british.mp3",
            TTS_TEST_DIR / "09_male_australian.mp3",
            TTS_TEST_DIR / "11_male_indian.mp3",
        ]

        results = []
        for audio_path in male_files:
            if not audio_path.exists():
                continue

            prediction = self._predict_file(audio_path, engine)
            results.append({
                "file": audio_path.name,
                "prediction": prediction.gender_prediction,
                "male_score": prediction.gender_scores["male"],
                "female_score": prediction.gender_scores["female"],
            })

            print(f"\n  {audio_path.name}: {prediction.gender_prediction} "
                  f"(M={prediction.gender_scores['male']:.2f}, F={prediction.gender_scores['female']:.2f})")

        # At least some files should exist
        assert len(results) > 0, "No male test files found"

        # Check that male score > female score for most files
        male_correct = sum(1 for r in results if r["male_score"] > r["female_score"])
        accuracy = male_correct / len(results)

        print(f"\n  Male detection accuracy: {accuracy:.0%} ({male_correct}/{len(results)})")

        # Expect at least 60% accuracy (acoustic method is limited)
        assert accuracy >= 0.6, f"Male detection accuracy too low: {accuracy:.0%}"

    def test_female_voices_detected(self, engine):
        """Female voice samples should be predicted as female."""
        female_files = [
            VOICE_TEST_DIR / "06_female_very_young.wav",
            VOICE_TEST_DIR / "07_female_young_adult.wav",
            VOICE_TEST_DIR / "08_female_middle_aged.wav",
            VOICE_TEST_DIR / "09_female_older.wav",
            TTS_TEST_DIR / "04_female_young_american.mp3",
            TTS_TEST_DIR / "05_female_adult_american.mp3",
            TTS_TEST_DIR / "08_female_british.mp3",
            TTS_TEST_DIR / "10_female_australian.mp3",
            TTS_TEST_DIR / "12_female_indian.mp3",
            TTS_TEST_DIR / "15_female_professional.mp3",
        ]

        results = []
        for audio_path in female_files:
            if not audio_path.exists():
                continue

            prediction = self._predict_file(audio_path, engine)
            results.append({
                "file": audio_path.name,
                "prediction": prediction.gender_prediction,
                "male_score": prediction.gender_scores["male"],
                "female_score": prediction.gender_scores["female"],
            })

            print(f"\n  {audio_path.name}: {prediction.gender_prediction} "
                  f"(M={prediction.gender_scores['male']:.2f}, F={prediction.gender_scores['female']:.2f})")

        assert len(results) > 0, "No female test files found"

        # Check that female score > male score for most files
        female_correct = sum(1 for r in results if r["female_score"] > r["male_score"])
        accuracy = female_correct / len(results)

        print(f"\n  Female detection accuracy: {accuracy:.0%} ({female_correct}/{len(results)})")

        assert accuracy >= 0.6, f"Female detection accuracy too low: {accuracy:.0%}"


# =============================================================================
# TEST: END-TO-END API WITH REAL AUDIO
# =============================================================================

class TestAPIWithRealAudio:
    """Test the complete API endpoint with real audio files."""

    def test_api_with_quality_samples(self, client):
        """Test API with quality test samples."""
        if not QUALITY_TEST_DIR.exists():
            pytest.skip("Quality test directory not found")

        files = get_audio_files(QUALITY_TEST_DIR)
        if not files:
            pytest.skip("No audio files found")

        print("\n=== API Tests with Quality Samples ===")

        for audio_path in files:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            start = time.perf_counter()
            response = client.post(
                "/analyze",
                files={"file": (audio_path.name, io.BytesIO(audio_bytes), "audio/wav")}
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response.status_code == 200, f"Failed for {audio_path.name}"
            data = response.json()

            print(f"\n  {audio_path.name}:")
            print(f"    Quality: {data['audio_quality']}")
            print(f"    Gender: {data['gender']['prediction']} ({data['gender']['confidence']:.2f})")
            print(f"    Age: {data['age_bracket']['prediction']} ({data['age_bracket']['confidence']:.2f})")
            print(f"    Latency: {elapsed_ms:.0f}ms (API: {data['processing_ms']}ms)")

            # All requests should complete under 500ms
            assert data['processing_ms'] < 500, f"Too slow: {data['processing_ms']}ms"

    def test_api_with_voice_samples(self, client):
        """Test API with voice test samples."""
        if not VOICE_TEST_DIR.exists():
            pytest.skip("Voice test directory not found")

        files = get_audio_files(VOICE_TEST_DIR)[:5]  # Test first 5
        if not files:
            pytest.skip("No audio files found")

        print("\n=== API Tests with Voice Samples ===")

        for audio_path in files:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            response = client.post(
                "/analyze",
                files={"file": (audio_path.name, io.BytesIO(audio_bytes), "audio/wav")}
            )

            assert response.status_code == 200, f"Failed for {audio_path.name}"
            data = response.json()

            print(f"\n  {audio_path.name}:")
            print(f"    Gender: {data['gender']['prediction']} ({data['gender']['confidence']:.2f})")
            print(f"    Age: {data['age_bracket']['prediction']}")

    def test_api_with_mp3_samples(self, client):
        """Test API with MP3 TTS samples."""
        if not TTS_TEST_DIR.exists():
            pytest.skip("TTS test directory not found")

        files = get_audio_files(TTS_TEST_DIR, extensions=(".mp3",))[:5]
        if not files:
            pytest.skip("No MP3 files found")

        print("\n=== API Tests with MP3 Samples ===")

        for audio_path in files:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            response = client.post(
                "/analyze",
                files={"file": (audio_path.name, io.BytesIO(audio_bytes), "audio/mpeg")}
            )

            assert response.status_code == 200, f"Failed for {audio_path.name}"
            data = response.json()

            print(f"\n  {audio_path.name}:")
            print(f"    Format: {data['_debug']['original_format']}")
            print(f"    Gender: {data['gender']['prediction']} ({data['gender']['confidence']:.2f})")
            print(f"    Latency: {data['processing_ms']}ms")

            # Verify MP3 was detected
            assert data['_debug']['original_format'] == 'mp3', \
                f"MP3 not detected: {data['_debug']['original_format']}"


# =============================================================================
# TEST: PERFORMANCE WITH REAL AUDIO
# =============================================================================

class TestPerformanceWithRealAudio:
    """Test latency requirements with real audio."""

    def test_all_files_under_500ms(self, client):
        """All audio files should process under 500ms (SLA requirement)."""
        all_dirs = [QUALITY_TEST_DIR, VOICE_TEST_DIR, TTS_TEST_DIR]
        all_files = []

        for d in all_dirs:
            if d.exists():
                all_files.extend(get_audio_files(d))

        if not all_files:
            pytest.skip("No test audio files found")

        print(f"\n=== Performance Test ({len(all_files)} files) ===")

        latencies = []
        failures = []

        for audio_path in all_files:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            content_type = "audio/mpeg" if audio_path.suffix == ".mp3" else "audio/wav"

            response = client.post(
                "/analyze",
                files={"file": (audio_path.name, io.BytesIO(audio_bytes), content_type)}
            )

            if response.status_code == 200:
                data = response.json()
                latency = data['processing_ms']
                latencies.append(latency)

                if latency > 500:
                    failures.append(f"{audio_path.name}: {latency}ms")

        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            min_latency = min(latencies)

            print(f"\n  Files processed: {len(latencies)}")
            print(f"  Avg latency: {avg_latency:.0f}ms")
            print(f"  Min latency: {min_latency:.0f}ms")
            print(f"  Max latency: {max_latency:.0f}ms")

            if failures:
                print(f"\n  SLA FAILURES (>500ms):")
                for f in failures:
                    print(f"    - {f}")

            # All files should be under 500ms
            assert len(failures) == 0, f"{len(failures)} files exceeded 500ms SLA"


# =============================================================================
# TEST: FULL PIPELINE SUMMARY
# =============================================================================

class TestFullPipelineSummary:
    """Generate a summary report of all test audio predictions."""

    def test_generate_prediction_report(self, client):
        """Generate a summary of predictions for all test files."""
        all_dirs = [
            ("Quality Tests", QUALITY_TEST_DIR),
            ("Voice Tests", VOICE_TEST_DIR),
            ("TTS Tests", TTS_TEST_DIR),
        ]

        print("\n" + "=" * 70)
        print("FULL PIPELINE PREDICTION REPORT")
        print("=" * 70)

        total_files = 0
        total_success = 0

        for section_name, directory in all_dirs:
            if not directory.exists():
                continue

            files = get_audio_files(directory)
            if not files:
                continue

            print(f"\n{'=' * 40}")
            print(f"{section_name}: {directory.name}/")
            print("=" * 40)
            print(f"{'File':<35} {'Quality':<12} {'Gender':<10} {'Age':<10} {'ms':>6}")
            print("-" * 70)

            for audio_path in files:
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()

                content_type = "audio/mpeg" if audio_path.suffix == ".mp3" else "audio/wav"

                response = client.post(
                    "/analyze",
                    files={"file": (audio_path.name, io.BytesIO(audio_bytes), content_type)}
                )

                total_files += 1

                if response.status_code == 200:
                    total_success += 1
                    data = response.json()
                    print(f"{audio_path.name:<35} "
                          f"{data['audio_quality']:<12} "
                          f"{data['gender']['prediction']:<10} "
                          f"{data['age_bracket']['prediction']:<10} "
                          f"{data['processing_ms']:>6.0f}")
                else:
                    print(f"{audio_path.name:<35} FAILED (HTTP {response.status_code})")

        print("\n" + "=" * 70)
        print(f"TOTAL: {total_success}/{total_files} files processed successfully")
        print("=" * 70)

        assert total_success == total_files, f"Some files failed: {total_files - total_success}"
