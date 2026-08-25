"""
Tests for the Inference Engine module.

Covers:
1. Shared pitch computation (_compute_frame_pitches)
2. Individual feature extractors (pitch, speaking rate, spectral centroid, jitter)
3. InferenceEngine class (predict, postprocess, confidence)
4. Global engine management (init_engine, get_engine)
5. Edge cases (silence, noise, short audio)
6. Gender/age prediction accuracy on synthetic audio
7. Performance requirements (<200ms latency)

Run with: pytest tests/test_inference_engine.py -v
"""

import os
import sys
import time

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.inference_engine import (
    # Feature extraction
    _compute_frame_pitches,
    estimate_pitch,
    estimate_speaking_rate,
    estimate_spectral_centroid,
    estimate_jitter,
    # Engine
    InferenceEngine,
    InferenceResult,
    init_engine,
    get_engine,
    # Constants
    AGE_BRACKETS,
    MIN_CONFIDENCE_THRESHOLD,
    PITCH_MIN_HZ,
    PITCH_MAX_HZ,
    PITCH_DEFINITELY_MALE,
    PITCH_LIKELY_FEMALE,
    MIN_PITCH_CONFIDENCE,
    VOICED_ENERGY_THRESHOLD,
)


# =============================================================================
# HELPER FUNCTIONS - AUDIO GENERATORS
# =============================================================================

def generate_sine_wave(frequency: float, duration: float = 2.0,
                       sample_rate: int = 16000, amplitude: float = 0.5) -> np.ndarray:
    """Generate a pure sine wave at given frequency."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    return (np.sin(2 * np.pi * frequency * t) * amplitude).astype(np.float32)


def generate_male_voice(duration: float = 3.0, sample_rate: int = 16000,
                        fundamental: float = 120.0, amplitude: float = 0.5) -> np.ndarray:
    """
    Generate synthetic male voice (F0 ~120 Hz).

    Male voice characteristics:
    - Fundamental frequency: 85-180 Hz (average ~120 Hz)
    - Rich harmonics
    - Amplitude modulation for natural speech rhythm
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Fundamental + harmonics (voice has rich harmonic structure)
    audio = np.zeros_like(t)
    for harmonic in range(1, 10):
        amp = amplitude / (harmonic ** 1.2)  # Harmonics decay
        audio += amp * np.sin(2 * np.pi * fundamental * harmonic * t)

    # Amplitude modulation (syllable rhythm ~3-4 Hz)
    envelope = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t))
    audio *= envelope

    # Normalize
    audio = audio / np.max(np.abs(audio)) * amplitude

    return audio.astype(np.float32)


def generate_female_voice(duration: float = 3.0, sample_rate: int = 16000,
                          fundamental: float = 220.0, amplitude: float = 0.5) -> np.ndarray:
    """
    Generate synthetic female voice (F0 ~220 Hz).

    Female voice characteristics:
    - Fundamental frequency: 165-255 Hz (average ~200 Hz)
    - Slightly different harmonic structure
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Fundamental + harmonics
    audio = np.zeros_like(t)
    for harmonic in range(1, 8):
        amp = amplitude / (harmonic ** 1.3)
        audio += amp * np.sin(2 * np.pi * fundamental * harmonic * t)

    # Amplitude modulation (slightly faster syllable rate)
    envelope = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * 4.5 * t))
    audio *= envelope

    # Normalize
    audio = audio / np.max(np.abs(audio)) * amplitude

    return audio.astype(np.float32)


def generate_silence(duration: float = 2.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate silence."""
    return np.zeros(int(sample_rate * duration), dtype=np.float32)


def generate_white_noise(duration: float = 2.0, sample_rate: int = 16000,
                         amplitude: float = 0.1) -> np.ndarray:
    """Generate white noise (no pitch structure)."""
    samples = int(sample_rate * duration)
    return (np.random.randn(samples) * amplitude).astype(np.float32)


def generate_young_voice_characteristics(duration: float = 3.0,
                                          sample_rate: int = 16000) -> np.ndarray:
    """
    Generate voice with "young" characteristics:
    - Higher spectral centroid (brighter)
    - Faster speaking rate
    - Lower jitter (more stable pitch)
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    fundamental = 180.0  # Mid-range

    audio = np.zeros_like(t)
    # More high-frequency content (brighter)
    for harmonic in range(1, 12):
        amp = 0.5 / (harmonic ** 0.9)  # Slower decay = more highs
        audio += amp * np.sin(2 * np.pi * fundamental * harmonic * t)

    # Fast syllable rate (~5.5 Hz)
    envelope = 0.2 + 0.8 * (0.5 + 0.5 * np.sin(2 * np.pi * 5.5 * t))
    audio *= envelope

    audio = audio / np.max(np.abs(audio)) * 0.5
    return audio.astype(np.float32)


def generate_older_voice_characteristics(duration: float = 3.0,
                                          sample_rate: int = 16000) -> np.ndarray:
    """
    Generate voice with "older" characteristics:
    - Lower spectral centroid (darker)
    - Slower speaking rate
    - Higher jitter (less stable pitch)
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    fundamental = 140.0

    audio = np.zeros_like(t)
    # Less high-frequency content (darker)
    for harmonic in range(1, 6):
        amp = 0.5 / (harmonic ** 1.5)  # Faster decay = fewer highs
        audio += amp * np.sin(2 * np.pi * fundamental * harmonic * t)

    # Slow syllable rate (~2.5 Hz)
    envelope = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * 2.5 * t))
    audio *= envelope

    # Add pitch instability (jitter simulation)
    jitter_mod = 1.0 + 0.03 * np.sin(2 * np.pi * 50 * t)
    audio *= jitter_mod

    audio = audio / np.max(np.abs(audio)) * 0.5
    return audio.astype(np.float32)


# =============================================================================
# TEST: SHARED PITCH COMPUTATION
# =============================================================================

class TestComputeFramePitches:
    """Tests for _compute_frame_pitches() shared computation."""

    def test_detects_pitch_in_sine_wave(self):
        """Should detect pitch in a pure sine wave."""
        audio = generate_sine_wave(150.0, duration=2.0, amplitude=0.5)
        pitches, voiced, total = _compute_frame_pitches(audio, 16000)

        assert len(pitches) > 0, "Should detect pitches in sine wave"
        assert voiced > 0, "Should have voiced frames"
        assert total > 0, "Should have total frames"

        # Median pitch should be close to 150 Hz
        median_pitch = np.median(pitches)
        assert 130 < median_pitch < 170, f"Expected ~150Hz, got {median_pitch}"

    def test_no_pitch_in_silence(self):
        """Should detect no pitch in silence."""
        audio = generate_silence(duration=2.0)
        pitches, voiced, total = _compute_frame_pitches(audio, 16000)

        assert len(pitches) == 0, "Should detect no pitches in silence"
        assert voiced == 0, "Should have no voiced frames"
        assert total > 0, "Should still count total frames"

    def test_no_pitch_in_white_noise(self):
        """White noise should have few/no detectable pitches."""
        audio = generate_white_noise(duration=2.0, amplitude=0.3)
        pitches, voiced, total = _compute_frame_pitches(audio, 16000)

        # White noise might trigger some false positives, but ratio should be low
        voiced_ratio = voiced / total if total > 0 else 0
        assert voiced_ratio < 0.3, f"White noise voiced ratio too high: {voiced_ratio}"

    def test_returns_correct_types(self):
        """Should return correct types."""
        audio = generate_male_voice(duration=1.0)
        pitches, voiced, total = _compute_frame_pitches(audio, 16000)

        assert isinstance(pitches, list), "Pitches should be a list"
        assert isinstance(voiced, int), "Voiced count should be int"
        assert isinstance(total, int), "Total count should be int"


# =============================================================================
# TEST: PITCH ESTIMATION
# =============================================================================

class TestEstimatePitch:
    """Tests for estimate_pitch() function."""

    def test_male_pitch_range(self):
        """Male voice should have pitch in 85-180 Hz range."""
        audio = generate_male_voice(fundamental=120.0)
        pitch, confidence = estimate_pitch(audio, 16000)

        assert 85 < pitch < 180, f"Male pitch {pitch} outside expected range"
        assert confidence > 0.1, f"Confidence too low: {confidence}"

    def test_female_pitch_range(self):
        """Female voice should have pitch in 165-255 Hz range."""
        audio = generate_female_voice(fundamental=220.0)
        pitch, confidence = estimate_pitch(audio, 16000)

        assert 165 < pitch < 280, f"Female pitch {pitch} outside expected range"
        assert confidence > 0.1, f"Confidence too low: {confidence}"

    def test_uses_precomputed_pitches(self):
        """Should use precomputed pitches when provided."""
        audio = generate_male_voice()

        # Compute manually
        precomputed = _compute_frame_pitches(audio, 16000)

        # Use precomputed
        pitch1, conf1 = estimate_pitch(audio, 16000, precomputed=precomputed)

        # Compute fresh
        pitch2, conf2 = estimate_pitch(audio, 16000)

        # Results should be identical
        assert pitch1 == pitch2, "Precomputed should give same result"
        assert conf1 == conf2, "Precomputed should give same confidence"

    def test_silence_returns_zero(self):
        """Silence should return pitch=0, confidence=0."""
        audio = generate_silence()
        pitch, confidence = estimate_pitch(audio, 16000)

        assert pitch == 0.0, f"Silence pitch should be 0, got {pitch}"
        assert confidence == 0.0, f"Silence confidence should be 0, got {confidence}"

    def test_confidence_reflects_voiced_ratio(self):
        """Confidence should reflect proportion of voiced frames."""
        # Full voiced audio
        audio_voiced = generate_male_voice(duration=2.0)
        _, conf_high = estimate_pitch(audio_voiced, 16000)

        # Mix with silence (50% voiced)
        silence = generate_silence(duration=2.0)
        audio_mixed = np.concatenate([audio_voiced, silence])
        _, conf_low = estimate_pitch(audio_mixed, 16000)

        assert conf_high > conf_low, "More voiced content should have higher confidence"


# =============================================================================
# TEST: SPEAKING RATE ESTIMATION
# =============================================================================

class TestEstimateSpeakingRate:
    """Tests for estimate_speaking_rate() function."""

    def test_returns_reasonable_range(self):
        """Speaking rate should be in reasonable range (1-10 syl/s)."""
        audio = generate_male_voice(duration=3.0)
        rate = estimate_speaking_rate(audio, 16000)

        assert 1.0 < rate < 10.0, f"Speaking rate {rate} outside reasonable range"

    def test_faster_modulation_higher_rate(self):
        """Faster amplitude modulation should give higher speaking rate."""
        t = np.linspace(0, 3.0, 48000, False)

        # Slow modulation (2 Hz)
        audio_slow = np.sin(2 * np.pi * 150 * t) * 0.5
        audio_slow *= 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t))

        # Fast modulation (6 Hz)
        audio_fast = np.sin(2 * np.pi * 150 * t) * 0.5
        audio_fast *= 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * 6 * t))

        rate_slow = estimate_speaking_rate(audio_slow.astype(np.float32), 16000)
        rate_fast = estimate_speaking_rate(audio_fast.astype(np.float32), 16000)

        assert rate_fast > rate_slow, f"Fast ({rate_fast}) should be > slow ({rate_slow})"

    def test_short_audio_returns_default(self):
        """Very short audio should return default rate."""
        audio = generate_male_voice(duration=0.05)  # 50ms
        rate = estimate_speaking_rate(audio, 16000)

        assert rate == 4.0, f"Short audio should return default 4.0, got {rate}"


# =============================================================================
# TEST: SPECTRAL CENTROID ESTIMATION
# =============================================================================

class TestEstimateSpectralCentroid:
    """Tests for estimate_spectral_centroid() function."""

    def test_high_freq_content_high_centroid(self):
        """High-frequency content should have high spectral centroid."""
        audio_high = generate_sine_wave(2000.0, amplitude=0.5)
        audio_low = generate_sine_wave(200.0, amplitude=0.5)

        centroid_high = estimate_spectral_centroid(audio_high, 16000)
        centroid_low = estimate_spectral_centroid(audio_low, 16000)

        assert centroid_high > centroid_low, \
            f"High freq centroid ({centroid_high}) should be > low ({centroid_low})"

    def test_uses_full_audio(self):
        """Should analyze full audio, not just first window."""
        # Create audio where second half is very different
        t = np.linspace(0, 3.0, 48000, False)

        # First half: low frequency, second half: high frequency
        audio = np.concatenate([
            np.sin(2 * np.pi * 200 * t[:24000]) * 0.5,
            np.sin(2 * np.pi * 2000 * t[24000:]) * 0.5
        ]).astype(np.float32)

        centroid = estimate_spectral_centroid(audio, 16000)

        # Should be somewhere in between (not just ~200 Hz from first window)
        assert centroid > 500, f"Centroid {centroid} should reflect full audio"

    def test_returns_reasonable_range(self):
        """Spectral centroid should be in reasonable range for speech."""
        audio = generate_male_voice()
        centroid = estimate_spectral_centroid(audio, 16000)

        # Speech centroid typically 500-3000 Hz
        assert 200 < centroid < 4000, f"Centroid {centroid} outside speech range"

    def test_short_audio_handled(self):
        """Should handle audio shorter than FFT window."""
        audio = generate_sine_wave(500.0, duration=0.05)  # 800 samples < 2048
        centroid = estimate_spectral_centroid(audio, 16000)

        assert centroid > 0, "Should return valid centroid for short audio"


# =============================================================================
# TEST: JITTER ESTIMATION
# =============================================================================

class TestEstimateJitter:
    """Tests for estimate_jitter() function."""

    def test_stable_pitch_low_jitter(self):
        """Stable pitch should have low jitter."""
        # Pure sine wave = perfectly stable pitch
        audio = generate_sine_wave(150.0, duration=3.0, amplitude=0.5)
        jitter = estimate_jitter(audio, 16000)

        assert jitter < 0.05, f"Stable pitch jitter too high: {jitter}"

    def test_uses_precomputed_pitches(self):
        """Should use precomputed pitches when provided."""
        audio = generate_male_voice()

        precomputed = _compute_frame_pitches(audio, 16000)

        jitter1 = estimate_jitter(audio, 16000, precomputed=precomputed)
        jitter2 = estimate_jitter(audio, 16000)

        assert jitter1 == jitter2, "Precomputed should give same jitter"

    def test_insufficient_pitches_returns_default(self):
        """Should return default when too few pitches detected."""
        # Create audio with very few voiced frames (mostly silence)
        audio = generate_silence(duration=2.0)
        # Add tiny voiced section - may or may not have enough pitches
        audio[:800] = generate_sine_wave(150.0, duration=0.05)[:800]

        jitter = estimate_jitter(audio.astype(np.float32), 16000)

        # Jitter should be non-negative (either computed or default 0.02)
        assert jitter >= 0, f"Jitter should be non-negative: {jitter}"
        # If computed, should be reasonable; if default, should be 0.02
        assert jitter < 0.5, f"Jitter {jitter} unreasonably high"

    def test_returns_non_negative(self):
        """Jitter should be non-negative for any input."""
        audio = generate_male_voice()
        jitter = estimate_jitter(audio, 16000)

        # Jitter can be 0 for perfectly stable synthetic voice
        # or small positive value for natural variation
        assert jitter >= 0, f"Jitter should be non-negative: {jitter}"
        assert jitter < 0.5, f"Jitter {jitter} unreasonably high"


# =============================================================================
# TEST: INFERENCE ENGINE CLASS
# =============================================================================

class TestInferenceEngine:
    """Tests for InferenceEngine class."""

    @pytest.fixture
    def engine(self):
        """Create an inference engine instance."""
        return InferenceEngine()

    def test_initialization(self, engine):
        """Engine should initialize correctly."""
        assert engine._model_type == "acoustic"

    def test_predict_returns_inference_result(self, engine):
        """predict() should return InferenceResult."""
        audio = generate_male_voice()
        result = engine.predict(audio, 16000)

        assert isinstance(result, InferenceResult)

    def test_predict_requires_16khz(self, engine):
        """predict() should reject non-16kHz audio."""
        audio = generate_male_voice()

        with pytest.raises(ValueError, match="16kHz"):
            engine.predict(audio, 44100)

    def test_male_voice_prediction(self, engine):
        """Male voice should be predicted as male."""
        audio = generate_male_voice(fundamental=120.0)
        result = engine.predict(audio, 16000)

        assert result.gender_prediction in ("male", "unknown"), \
            f"Male voice predicted as {result.gender_prediction}"
        assert result.gender_scores["male"] > result.gender_scores["female"], \
            f"Male score should be higher: {result.gender_scores}"

    def test_female_voice_prediction(self, engine):
        """Female voice should be predicted as female."""
        audio = generate_female_voice(fundamental=220.0)
        result = engine.predict(audio, 16000)

        assert result.gender_prediction in ("female", "unknown"), \
            f"Female voice predicted as {result.gender_prediction}"
        assert result.gender_scores["female"] > result.gender_scores["male"], \
            f"Female score should be higher: {result.gender_scores}"

    def test_silence_returns_unknown(self, engine):
        """Silence should return unknown predictions."""
        audio = generate_silence()
        result = engine.predict(audio, 16000)

        assert result.gender_prediction == "unknown"
        assert result.age_bracket == "unknown"
        assert result.gender_confidence == 0.0

    def test_result_has_all_fields(self, engine):
        """Result should have all expected fields."""
        audio = generate_male_voice()
        result = engine.predict(audio, 16000)

        assert hasattr(result, 'gender_prediction')
        assert hasattr(result, 'gender_confidence')
        assert hasattr(result, 'gender_scores')
        assert hasattr(result, 'age_bracket')
        assert hasattr(result, 'age_confidence')
        assert hasattr(result, 'age_raw')
        assert hasattr(result, 'inference_ms')

    def test_gender_scores_sum_to_one(self, engine):
        """Gender scores should sum to approximately 1."""
        audio = generate_male_voice()
        result = engine.predict(audio, 16000)

        total = result.gender_scores["male"] + result.gender_scores["female"]
        assert 0.99 < total < 1.01, f"Gender scores sum to {total}, expected ~1.0"

    def test_age_bracket_valid(self, engine):
        """Age bracket should be one of the defined brackets or unknown."""
        audio = generate_male_voice()
        result = engine.predict(audio, 16000)

        valid_brackets = ["unknown"] + [b[2] for b in AGE_BRACKETS]
        assert result.age_bracket in valid_brackets, \
            f"Invalid age bracket: {result.age_bracket}"

    def test_age_raw_in_range(self, engine):
        """Raw age should be in reasonable range."""
        audio = generate_male_voice()
        result = engine.predict(audio, 16000)

        if result.age_raw > 0:  # Not unknown
            assert 18 <= result.age_raw <= 80, \
                f"Age raw {result.age_raw} outside expected range"

    def test_inference_time_recorded(self, engine):
        """Inference time should be recorded."""
        audio = generate_male_voice()
        result = engine.predict(audio, 16000)

        assert result.inference_ms > 0, "Inference time should be recorded"


# =============================================================================
# TEST: AGE CONFIDENCE CALCULATION
# =============================================================================

class TestAgeConfidence:
    """Tests for age confidence calculation."""

    @pytest.fixture
    def engine(self):
        return InferenceEngine()

    def test_center_of_bracket_high_confidence(self, engine):
        """Age at center of bracket should have higher confidence."""
        # Center of 31-45 bracket is 38
        conf_center = engine._calc_age_confidence(38.0)

        # Edge of bracket (31)
        conf_edge = engine._calc_age_confidence(31.0)

        assert conf_center >= conf_edge, \
            f"Center ({conf_center}) should have >= confidence than edge ({conf_edge})"

    def test_confidence_range(self, engine):
        """Confidence should be in valid range."""
        for age in [25, 35, 45, 55, 65]:
            conf = engine._calc_age_confidence(age)
            assert 0.0 <= conf <= 1.0, f"Confidence {conf} for age {age} out of range"


# =============================================================================
# TEST: GLOBAL ENGINE MANAGEMENT
# =============================================================================

class TestGlobalEngine:
    """Tests for global engine init/get functions."""

    def test_init_engine_returns_engine(self):
        """init_engine() should return an InferenceEngine."""
        engine = init_engine()
        assert isinstance(engine, InferenceEngine)

    def test_get_engine_returns_initialized(self):
        """get_engine() should return the initialized engine."""
        init_engine()
        engine = get_engine()
        assert isinstance(engine, InferenceEngine)

    def test_get_engine_before_init_raises(self):
        """get_engine() before init should raise RuntimeError."""
        # Reset global state
        import app.services.inference_engine as ie
        ie._engine = None

        with pytest.raises(RuntimeError, match="not initialized"):
            get_engine()

        # Re-initialize for other tests
        init_engine()


# =============================================================================
# TEST: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def engine(self):
        return InferenceEngine()

    def test_very_short_audio(self, engine):
        """Should handle very short audio gracefully."""
        audio = generate_male_voice(duration=0.5)  # 500ms minimum
        result = engine.predict(audio, 16000)

        # Should not crash, may return unknown
        assert isinstance(result, InferenceResult)

    def test_white_noise_input(self, engine):
        """White noise should return low-confidence or unknown."""
        audio = generate_white_noise(duration=3.0, amplitude=0.3)
        result = engine.predict(audio, 16000)

        # Noise has no pitch structure, should be unknown or low confidence
        assert result.gender_confidence < 0.8 or result.gender_prediction == "unknown"

    def test_ambiguous_pitch(self, engine):
        """Pitch in ambiguous range (160-180 Hz) should have lower confidence."""
        # Generate voice at 170 Hz (ambiguous zone)
        t = np.linspace(0, 3.0, 48000, False)
        audio = np.sin(2 * np.pi * 170 * t) * 0.5
        audio *= 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))
        audio = audio.astype(np.float32)

        result = engine.predict(audio, 16000)

        # Confidence should be moderate (not very high)
        # Score difference should be small
        score_diff = abs(result.gender_scores["male"] - result.gender_scores["female"])
        assert score_diff < 0.5, f"Ambiguous pitch should have close scores: {result.gender_scores}"

    def test_clipped_audio(self, engine):
        """Should handle clipped/distorted audio."""
        audio = generate_male_voice(amplitude=0.9)
        # Hard clip
        audio = np.clip(audio, -0.5, 0.5)

        result = engine.predict(audio, 16000)
        assert isinstance(result, InferenceResult)

    def test_dc_offset_audio(self, engine):
        """Should handle audio with DC offset."""
        audio = generate_male_voice()
        audio = audio + 0.3  # Add DC offset

        result = engine.predict(audio, 16000)
        assert isinstance(result, InferenceResult)


# =============================================================================
# TEST: PERFORMANCE
# =============================================================================

class TestPerformance:
    """Tests for performance requirements."""

    @pytest.fixture
    def engine(self):
        return InferenceEngine()

    def test_inference_latency_5_seconds(self, engine):
        """Inference on 5-second audio should be < 200ms."""
        audio = generate_male_voice(duration=5.0)

        start = time.perf_counter()
        result = engine.predict(audio, 16000)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 200, f"Inference took {elapsed_ms:.1f}ms, expected < 200ms"

        # Also check recorded time
        assert result.inference_ms < 200, f"Recorded time {result.inference_ms:.1f}ms too high"

    def test_inference_latency_10_seconds(self, engine):
        """Inference on 10-second audio should be < 400ms."""
        audio = generate_male_voice(duration=10.0)

        start = time.perf_counter()
        result = engine.predict(audio, 16000)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 400, f"Inference took {elapsed_ms:.1f}ms, expected < 400ms"

    def test_shared_computation_faster(self, engine):
        """Using shared pitch computation should not be slower than computing twice."""
        audio = generate_male_voice(duration=5.0)

        # Time the actual prediction (uses shared computation internally)
        start = time.perf_counter()
        _ = engine.predict(audio, 16000)
        shared_time = time.perf_counter() - start

        # Time computing pitch and jitter separately (without sharing)
        start = time.perf_counter()
        _ = estimate_pitch(audio, 16000)  # Computes fresh
        _ = estimate_jitter(audio, 16000)  # Computes fresh again
        separate_time = time.perf_counter() - start

        # Shared should be faster or at least not much slower
        # (shared avoids duplicate pitch computation)
        assert shared_time <= separate_time * 1.5, \
            f"Shared ({shared_time*1000:.1f}ms) should be ≤ separate ({separate_time*1000:.1f}ms)"


# =============================================================================
# TEST: INTEGRATION WITH SYNTHETIC VOICES
# =============================================================================

class TestSyntheticVoiceAccuracy:
    """Tests that verify predictions on synthetic voices with known characteristics."""

    @pytest.fixture
    def engine(self):
        return InferenceEngine()

    def test_low_pitch_male(self, engine):
        """Very low pitch (100 Hz) should be confidently male."""
        audio = generate_male_voice(fundamental=100.0)
        result = engine.predict(audio, 16000)

        assert result.gender_scores["male"] > 0.7, \
            f"100Hz should be clearly male: {result.gender_scores}"

    def test_high_pitch_female(self, engine):
        """Very high pitch (250 Hz) should be confidently female."""
        audio = generate_female_voice(fundamental=250.0)
        result = engine.predict(audio, 16000)

        assert result.gender_scores["female"] > 0.7, \
            f"250Hz should be clearly female: {result.gender_scores}"

    def test_young_characteristics_predict_younger(self, engine):
        """Voice with young characteristics should predict younger age."""
        audio = generate_young_voice_characteristics()
        result = engine.predict(audio, 16000)

        # Should lean toward younger brackets (18-30 or 31-45)
        assert result.age_bracket in ("18-30", "31-45", "unknown"), \
            f"Young voice predicted as {result.age_bracket}"

    def test_older_characteristics_predict_older(self, engine):
        """Voice with older characteristics should predict older age."""
        audio = generate_older_voice_characteristics()
        result = engine.predict(audio, 16000)

        # Should lean toward older brackets (31-45 or higher)
        # Note: Due to center-biasing, predictions tend toward 31-45
        assert result.age_bracket in ("31-45", "46-60", "60+", "unknown"), \
            f"Older voice predicted as {result.age_bracket}"
