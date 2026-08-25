"""
Quality Analyzer Module - Dialflo Voice Demographic Service

This module assesses audio quality to determine if it's suitable for
ML inference. It acts as a gatekeeper that can:
- Pass good audio through with full confidence
- Flag degraded audio (proceed but lower confidence)
- Short-circuit insufficient audio (skip inference, save latency)

METRICS CALCULATED:
    - RMS Energy: Overall signal strength (speech band filtered)
    - SNR Estimate: Speech-to-noise ratio (dB)
    - Speech Ratio: Fraction of audio containing speech
    - Clipping: Distortion from too-loud recording
    - Low-Freq Noise: Truck engine / HVAC detection

LOGISTICS CONTEXT:
    Common noise sources: truck engines (50-200Hz), road noise (broadband),
    wind (20-500Hz), warehouse reverb, compressed VoIP codecs.

FIXES IN v3:
    1. 80Hz cutoff (not 200Hz) - preserves male voice fundamentals (85-180Hz)
    2. Filter ONCE in analyze_quality, pass filtered audio to metrics
    3. Use SOS + sosfilt (single-pass, numerically stable) instead of filtfilt

TARGET LATENCY: <10ms for 5-second audio
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np
from scipy import signal

logger = logging.getLogger("dialflo-backend")


# =============================================================================
# CONFIGURATION
# =============================================================================

class AudioQuality(Enum):
    """Audio quality classification levels."""
    GOOD = "good"
    DEGRADED = "degraded"
    INSUFFICIENT = "insufficient"


@dataclass
class QualityResult:
    """Container for quality analysis results."""
    quality: AudioQuality
    rms_energy: float
    snr_estimate_db: float
    speech_ratio: float
    clipping_ratio: float
    low_freq_ratio: float
    analysis_ms: float

    def to_dict(self) -> dict:
        return {
            "quality": self.quality.value,
            "rms_energy": round(self.rms_energy, 4),
            "snr_estimate_db": round(self.snr_estimate_db, 1),
            "speech_ratio": round(self.speech_ratio, 2),
            "clipping_ratio": round(self.clipping_ratio, 4),
            "low_freq_ratio": round(self.low_freq_ratio, 2),
            "analysis_ms": round(self.analysis_ms, 1)
        }


# Thresholds for quality classification
# Tuned for speech audio in noisy logistics environments

# RMS Energy thresholds (on HIGH-PASS FILTERED audio)
RMS_GOOD = 0.01        # Clear speech typically > 0.01
RMS_DEGRADED = 0.003   # Very quiet but audible
RMS_SILENCE = 0.001    # Essentially silence

# SNR thresholds (in dB)
SNR_GOOD = 15.0        # Clear speech, minimal background noise
SNR_DEGRADED = 5.0     # Noisy but speech is distinguishable
SNR_INSUFFICIENT = 0.0 # Noise dominates, speech barely audible

# Speech ratio thresholds (fraction of frames with speech activity)
SPEECH_RATIO_GOOD = 0.3       # At least 30% of audio has speech
SPEECH_RATIO_DEGRADED = 0.1   # At least 10% has speech
SPEECH_RATIO_INSUFFICIENT = 0.05

# Clipping thresholds (fraction of samples at max amplitude)
CLIPPING_ACCEPTABLE = 0.01    # <1% clipping is OK
CLIPPING_SEVERE = 0.10        # >10% clipping is problematic

# Low-frequency noise thresholds (truck engine detection)
LOW_FREQ_ACCEPTABLE = 0.4     # <40% energy below 80Hz is normal
LOW_FREQ_EXCESSIVE = 0.7      # >70% = mostly low-freq rumble


# =============================================================================
# HIGH-PASS FILTER (removes truck rumble, preserves male voices)
# =============================================================================

# CUTOFF: 80Hz (not 200Hz!)
# - Truck engines: 30-80Hz fundamental harmonics
# - Male voice fundamental: 85-180Hz
# - Female voice fundamental: 165-255Hz
# 80Hz cutoff blocks engine rumble while preserving ALL human speech!

HP_CUTOFF_HZ = 80

# Pre-compute SOS coefficients for 16kHz audio
# Using SOS (Second-Order Sections) for numerical stability
# 4th order Butterworth high-pass
_HP_FILTER_SOS_16K = signal.butter(4, HP_CUTOFF_HZ, btype='high', fs=16000, output='sos')


def _apply_highpass_filter(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Apply high-pass filter to remove low-frequency noise (truck rumble).

    DESIGN CHOICES:
    - 80Hz cutoff: Blocks engine rumble (30-80Hz) while preserving
      male voice fundamentals (85-180Hz)
    - SOS format: Numerically stable for higher filter orders
    - sosfilt: Single forward pass (not filtfilt's forward+backward)
      = 2x faster, suitable for real-time

    Returns:
        Filtered audio with low frequencies removed
    """
    if len(audio) < 100:
        return audio

    try:
        if sample_rate == 16000:
            # Use pre-computed SOS coefficients
            filtered = signal.sosfilt(_HP_FILTER_SOS_16K, audio)
        else:
            # Compute coefficients for different sample rate
            sos = signal.butter(4, HP_CUTOFF_HZ, btype='high', fs=sample_rate, output='sos')
            filtered = signal.sosfilt(sos, audio)
        return filtered.astype(np.float32)
    except Exception:
        # If filtering fails, return original
        return audio


# =============================================================================
# METRIC CALCULATIONS
# =============================================================================

def calculate_rms_energy(filtered_audio: np.ndarray) -> float:
    """
    Calculate RMS energy on PRE-FILTERED audio.

    NOTE: Expects already high-pass filtered audio from analyze_quality().
    Do NOT call _apply_highpass_filter here - it's done once upstream.

    Returns:
        RMS value in range [0, 1] for normalized audio
    """
    return float(np.sqrt(np.mean(filtered_audio ** 2)))


def calculate_snr_estimate(filtered_audio: np.ndarray, sample_rate: int) -> float:
    """
    Estimate Signal-to-Noise Ratio on PRE-FILTERED audio.

    NOTE: Expects already high-pass filtered audio from analyze_quality().

    Strategy: Compare energy of high-energy frames (likely speech)
    to low-energy frames (likely silence/noise).

    Returns:
        SNR estimate in decibels (dB)
    """
    # Frame-based approach: split into 30ms frames
    frame_length = int(sample_rate * 0.03)  # 30ms frames

    if len(filtered_audio) < frame_length * 4:
        # Too short for reliable estimation
        rms = np.sqrt(np.mean(filtered_audio ** 2))
        return 20.0 if rms > 0.01 else 5.0

    # Calculate energy per frame
    num_frames = len(filtered_audio) // frame_length
    frames = filtered_audio[:num_frames * frame_length].reshape(num_frames, frame_length)
    frame_energies = np.mean(frames ** 2, axis=1)

    # Sort frames by energy
    sorted_energies = np.sort(frame_energies)

    # Signal estimate: top 30% of frames (likely speech)
    # Noise estimate: bottom 20% of frames (likely silence/noise)
    top_30_pct = int(num_frames * 0.7)
    bottom_20_pct = int(num_frames * 0.2)

    signal_energy = np.mean(sorted_energies[top_30_pct:])
    noise_energy = np.mean(sorted_energies[:max(bottom_20_pct, 1)])

    # Avoid division by zero
    if noise_energy < 1e-12:
        return 40.0  # Very clean, essentially no noise floor

    snr_linear = signal_energy / noise_energy
    snr_db = 10 * np.log10(max(snr_linear, 1e-10))

    # Clamp to reasonable range
    return float(np.clip(snr_db, -10, 50))


def calculate_speech_ratio(filtered_audio: np.ndarray, sample_rate: int,
                           frame_length_ms: int = 30) -> float:
    """
    Estimate the fraction of audio containing speech using PERCENTILE-based VAD.

    NOTE: Expects already high-pass filtered audio from analyze_quality().

    Uses percentile range to find frames significantly above
    the noise floor, regardless of absolute noise level.

    Returns:
        Fraction of frames containing speech [0, 1]
    """
    frame_length = int(sample_rate * frame_length_ms / 1000)

    if len(filtered_audio) < frame_length:
        rms = np.sqrt(np.mean(filtered_audio ** 2))
        return 1.0 if rms > RMS_DEGRADED else 0.0

    # Split into frames
    num_frames = len(filtered_audio) // frame_length
    frames = filtered_audio[:num_frames * frame_length].reshape(num_frames, frame_length)

    # Calculate energy per frame
    frame_energies = np.sqrt(np.mean(frames ** 2, axis=1))

    # PERCENTILE-BASED THRESHOLD
    # p10 = noise floor estimate
    # p90 = peak energy (speech + noise)
    # Threshold = p10 + 0.3 * (p90 - p10)
    p10 = np.percentile(frame_energies, 10)
    p90 = np.percentile(frame_energies, 90)

    dynamic_range = p90 - p10

    if dynamic_range < 1e-6:
        # No dynamic range = constant level = likely just noise
        return 0.0

    # Threshold at 30% above the noise floor
    threshold = p10 + dynamic_range * 0.3
    threshold = max(threshold, RMS_SILENCE)

    # Count frames above threshold (likely speech)
    speech_frames = np.sum(frame_energies > threshold)

    return float(speech_frames / num_frames)


def calculate_clipping_ratio(audio: np.ndarray,
                              clip_threshold: float = 0.99) -> float:
    """
    Calculate the ratio of samples that are clipped (at max amplitude).

    NOTE: Uses ORIGINAL audio (not filtered) - clipping happens before filtering.

    Returns:
        Fraction of samples that are clipped [0, 1]
    """
    clipped_samples = np.sum(np.abs(audio) >= clip_threshold)
    return float(clipped_samples / len(audio))


def calculate_low_frequency_noise(audio: np.ndarray, sample_rate: int,
                                   low_freq_cutoff: int = 80) -> float:
    """
    Detect excessive low-frequency noise (truck engines, HVAC, road rumble).

    NOTE: Uses ORIGINAL audio (not filtered) - we're measuring low-freq content.

    High ratio (>0.7) indicates most energy is below 80Hz = likely engine noise.

    Returns:
        Ratio of low-frequency energy to total energy [0, 1]
    """
    nperseg = min(1024, len(audio) // 4)
    if nperseg < 256:
        return 0.0

    try:
        freqs, psd = signal.welch(audio, sample_rate, nperseg=nperseg)
    except Exception:
        return 0.0

    low_freq_mask = freqs < low_freq_cutoff
    low_freq_power = np.sum(psd[low_freq_mask])
    total_power = np.sum(psd)

    if total_power < 1e-10:
        return 0.0

    return float(low_freq_power / total_power)


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_quality(audio: np.ndarray, sample_rate: int = 16000) -> QualityResult:
    """
    Analyze audio quality and classify as good/degraded/insufficient.

    OPTIMIZATION: High-pass filter is applied ONCE here, then the filtered
    audio is passed to metric functions. This avoids triple-filtering.

    Args:
        audio: 16kHz mono float32 numpy array from decoder
        sample_rate: Sample rate (default 16000)

    Returns:
        QualityResult with metrics and classification
    """
    start_time = time.perf_counter()

    # === STEP 1: Apply high-pass filter ONCE ===
    # This removes <80Hz (truck rumble) while preserving male voices (85Hz+)
    filtered_audio = _apply_highpass_filter(audio, sample_rate)

    # === STEP 2: Calculate metrics ===
    # RMS, SNR, speech_ratio use FILTERED audio (speech frequencies only)
    # Clipping, low_freq_ratio use ORIGINAL audio
    rms_energy = calculate_rms_energy(filtered_audio)
    snr_estimate = calculate_snr_estimate(filtered_audio, sample_rate)
    speech_ratio = calculate_speech_ratio(filtered_audio, sample_rate)
    clipping_ratio = calculate_clipping_ratio(audio)  # Original audio
    low_freq_ratio = calculate_low_frequency_noise(audio, sample_rate)  # Original audio

    # Log metrics for debugging/tuning
    logger.debug(
        f"Quality metrics: RMS={rms_energy:.4f}, SNR={snr_estimate:.1f}dB, "
        f"speech={speech_ratio:.2f}, clipping={clipping_ratio:.4f}, "
        f"low_freq={low_freq_ratio:.2f}"
    )

    # === STEP 3: Classification ===
    quality = _classify_quality(
        rms_energy, snr_estimate, speech_ratio, clipping_ratio, low_freq_ratio
    )

    analysis_ms = (time.perf_counter() - start_time) * 1000

    result = QualityResult(
        quality=quality,
        rms_energy=rms_energy,
        snr_estimate_db=snr_estimate,
        speech_ratio=speech_ratio,
        clipping_ratio=clipping_ratio,
        low_freq_ratio=low_freq_ratio,
        analysis_ms=analysis_ms
    )

    logger.info(f"Quality analysis: {quality.value} (took {analysis_ms:.1f}ms)")

    return result


def _classify_quality(rms: float, snr: float, speech_ratio: float,
                      clipping: float, low_freq_ratio: float) -> AudioQuality:
    """
    Classify audio quality based on calculated metrics.

    Uses a hierarchical decision tree:
    1. First check for deal-breakers (silence, severe clipping, excessive noise)
    2. Then check for degraded conditions
    3. Default to good if no issues found
    """

    # === INSUFFICIENT: Deal-breakers ===

    # Near silence (after high-pass filtering)
    if rms < RMS_SILENCE:
        logger.debug("Insufficient: near silence")
        return AudioQuality.INSUFFICIENT

    # Severe clipping (distortion)
    if clipping > CLIPPING_SEVERE:
        logger.debug("Insufficient: severe clipping")
        return AudioQuality.INSUFFICIENT

    # No speech detected
    if speech_ratio < SPEECH_RATIO_INSUFFICIENT:
        logger.debug("Insufficient: no speech detected")
        return AudioQuality.INSUFFICIENT

    # Extremely noisy (SNR very negative = noise dominates)
    if snr < SNR_INSUFFICIENT:
        logger.debug("Insufficient: noise dominates signal")
        return AudioQuality.INSUFFICIENT

    # Excessive low-frequency noise (truck engine drowns out speech)
    if low_freq_ratio > LOW_FREQ_EXCESSIVE:
        logger.debug("Insufficient: excessive low-frequency noise (engine?)")
        return AudioQuality.INSUFFICIENT

    # === DEGRADED: Usable but not ideal ===

    # Quiet audio (speech band)
    if rms < RMS_GOOD:
        logger.debug("Degraded: quiet audio")
        return AudioQuality.DEGRADED

    # Noisy
    if snr < SNR_GOOD:
        logger.debug("Degraded: noisy audio")
        return AudioQuality.DEGRADED

    # Low speech content
    if speech_ratio < SPEECH_RATIO_GOOD:
        logger.debug("Degraded: low speech content")
        return AudioQuality.DEGRADED

    # Some clipping
    if clipping > CLIPPING_ACCEPTABLE:
        logger.debug("Degraded: some clipping detected")
        return AudioQuality.DEGRADED

    # Significant low-frequency noise (but not overwhelming)
    if low_freq_ratio > LOW_FREQ_ACCEPTABLE:
        logger.debug("Degraded: significant low-frequency noise")
        return AudioQuality.DEGRADED

    # === GOOD: All checks passed ===
    return AudioQuality.GOOD


# =============================================================================
# CONFIDENCE ADJUSTMENT
# =============================================================================

def get_confidence_multiplier(quality: AudioQuality) -> float:
    """
    Get confidence multiplier based on audio quality.

    ML predictions should be scaled by this factor to reflect
    uncertainty from audio quality issues.

    Returns:
        Multiplier in range [0, 1]
    """
    multipliers = {
        AudioQuality.GOOD: 1.0,
        AudioQuality.DEGRADED: 0.7,
        AudioQuality.INSUFFICIENT: 0.0
    }
    return multipliers.get(quality, 0.5)
