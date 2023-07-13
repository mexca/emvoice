"""Energy-related voice features."""

import logging

import librosa
import numpy as np
from scipy.signal import find_peaks

from emvoice.frames import BaseFrames
from emvoice.spectral import SpecFrames


class HnrFrames(BaseFrames):
    """Estimate and store harmonics-to-noise ratios (HNRs).

    Parameters
    ----------
    frames: numpy.ndarray
        HNR frames in dB with shape (num_frames,).
    lower: float
        Lower fundamental frequency limit for choosing pitch candidates.
    rel_silence_threshold: float
        Relative threshold for treating signal frames as silent.

    Notes
    -----
    Estimate the HNR for each signal frame with ``np.max(np.abs(frames), axis=1) > rel_silence_threshold*np.max(np.abs(frames))`` by:

    1. Compute the autocorrelation function (ACF) using the short-term Fourier transform (STFT).
    2. Find the lags of peaks in the ACF excluding the zero-th lag.
    3. Filter out peaks that correspond to pitch candidates below `lower` and above the Nyquist frequency.
    4. Compute the harmonic component `R0` as the highest of the remaining peaks divided by the ACF at lag zero.
    5. Compute the HNR as `R0/(1-R0)` and convert to dB.

    """

    def __init__(
        self,
        frames: np.ndarray,
        sr: int,
        frame_len: int,
        hop_len: int,
        center: bool,
        pad_mode: str,
        lower: float,
        rel_silence_threshold,
    ):
        self.logger = logging.getLogger("emvoice.frequency.HnrFrames")
        self.lower = lower
        self.rel_silence_threshold = rel_silence_threshold
        super().__init__(frames, sr, frame_len, hop_len, center, pad_mode)

    @classmethod
    def from_frames(
        cls,
        sig_frames_obj: BaseFrames,
        lower: float = 75.0,
        rel_silence_threshold: float = 0.1,
    ):
        """Estimate the HNR from signal frames.

        Parameters
        ----------
        sig_frames_obj: BaseFrames
            Signal frames object.
        lower: float, default = 75.0
            Lower fundamental frequency limit for choosing pitch candidates.
        rel_silence_threshold: float, default = 0.1
            Relative threshold for treating signal frames as silent.

        """
        auto_cor = librosa.autocorrelate(sig_frames_obj.frames)
        harmonic_strength = np.apply_along_axis(
            cls._find_max_peak,
            1,
            auto_cor[:, 1:],
            sr=sig_frames_obj.sr,
            lower=lower,
        )
        harmonic_comp = harmonic_strength / auto_cor[:, 0]
        hnr = harmonic_comp / (1 - harmonic_comp)
        silence_mask = np.max(
            np.abs(sig_frames_obj.frames), axis=1
        ) > rel_silence_threshold * np.max(np.abs(sig_frames_obj.frames))
        hnr[np.logical_or(~silence_mask, hnr <= 0)] = np.nan
        hnr_db = librosa.power_to_db(hnr)  # HNR is on power scale
        return cls(
            hnr_db,
            sig_frames_obj.sr,
            sig_frames_obj.frame_len,
            sig_frames_obj.hop_len,
            sig_frames_obj.center,
            sig_frames_obj.pad_mode,
            lower,
            rel_silence_threshold,
        )

    @staticmethod
    def _find_max_peak(auto_cor: np.ndarray, sr: int, lower: float) -> float:
        if np.all(np.isnan(auto_cor)):
            return np.nan

        auto_cor_peak_lags = find_peaks(auto_cor)[0]
        auto_cor_peaks = auto_cor[auto_cor_peak_lags]
        auto_cor_peak_periods = 1 / auto_cor_peak_lags * sr
        auto_cor_peaks_voiced = auto_cor_peaks[
            np.logical_and(
                auto_cor_peak_periods > lower, auto_cor_peak_periods < sr / 2
            )
        ]

        if len(auto_cor_peaks_voiced) == 0:
            return np.nan

        auto_cor_max_peak_lag = np.argmax(auto_cor_peaks_voiced)

        return auto_cor_peaks_voiced[auto_cor_max_peak_lag]


class RmsEnergyFrames(SpecFrames):
    """Calculate and store the root mean squared (RMS) energy.

    Parameters
    ---------
    frames: numpy.ndarray
        RMS energy frames in dB with shape (num_frames,).

    """

    @classmethod
    def from_spec_frames(cls, spec_frames_obj: SpecFrames):
        """Calculate the RMS energy from spectrogram frames.

        Parameters
        ----------
        spec_frames_obj: SpecFrames
            Spectrogram frames object.

        """
        rms_frames = librosa.amplitude_to_db(
            librosa.feature.rms(  # to dB
                S=np.abs(spec_frames_obj.frames).T,
                frame_length=spec_frames_obj.frame_len,
                hop_length=spec_frames_obj.hop_len,
                center=spec_frames_obj.center,
                pad_mode=spec_frames_obj.pad_mode,
            )
        )

        return cls(
            rms_frames.squeeze(),
            spec_frames_obj.sr,
            spec_frames_obj.window,
            spec_frames_obj.frame_len,
            spec_frames_obj.hop_len,
            spec_frames_obj.center,
            spec_frames_obj.pad_mode,
        )
