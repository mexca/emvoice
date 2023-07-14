"""Formant-related voice features."""

import logging
import math
from copy import copy
from typing import List, Optional, Tuple, Union

import librosa
import numpy as np
from scipy.signal.windows import get_window

from emvoice.frames import BaseFrames
from emvoice.pitch import PitchFrames, PitchHarmonicsFrames


class FormantFrames(BaseFrames):
    """Estimate and store formant frames.

    Parameters
    ----------
    frames: list
        Formant frames. Each frame contains a list of tuples for each formant, where the first item
        is the central frequency and the second the bandwidth.
    max_formants: int, default=5
        The maximum number of formants that were extracted.
    lower: float, default=50.0
        Lower limit for formant frequencies (in Hz).
    upper: float, default=5450.0
        Upper limit for formant frequencies (in Hz).
    preemphasis_from: float, default=50.0
        Starting value for the applied preemphasis function.
    window: str
        Window function that was applied before formant estimation.

    Notes
    -----
    See the :ref:`Algorithms section <Formant frequencies and amplitudes>` for details.

    """

    def __init__(
        self,
        frames: List,
        sr: int,
        frame_len: int,
        hop_len: int,
        center: bool = True,
        pad_mode: str = "constant",
        max_formants: int = 5,
        lower: float = 50.0,
        upper: float = 5450.0,
        preemphasis_from: Optional[float] = 50.0,
        window: Optional[Union[str, float, Tuple]] = "praat_gaussian",
    ) -> None:
        self.logger = logging.getLogger("emvoice.frequency.FormantFrames")
        self.max_formants = max_formants
        self.lower = lower
        self.upper = upper
        self.preemphasis_from = preemphasis_from
        self.window = window
        super().__init__(frames, sr, frame_len, hop_len, center, pad_mode)

    @property
    def idx(self) -> np.ndarray:
        if self._idx is None:
            self._idx = np.arange(len(self.frames))
        return self._idx

    @staticmethod
    def _praat_gauss_window(frame_len: int):
        # This is the Gaussian window that is used in Praat for formant estimation
        # See: https://github.com/YannickJadoul/Parselmouth/blob/master/praat/fon/Sound_to_Formant.cpp
        sample_idx = np.arange(frame_len) + 1
        idx_mid = 0.5 * (frame_len + 1)
        edge = np.exp(-12.0)
        return (
            np.exp(-48.0 * (sample_idx - idx_mid) ** 2 / (frame_len) ** 2)
            - edge
        ) / (1.0 - edge)

    @classmethod
    def from_frames(
        cls,
        sig_frames_obj: BaseFrames,
        max_formants: int = 5,
        lower: float = 50.0,
        upper: float = 5450.0,
        preemphasis_from: Optional[float] = 50.0,
        window: Optional[Union[str, float, Tuple]] = "praat_gaussian",
    ):
        """Extract formants from signal frames.

        Parameters
        ----------
        sig_frames_obj: BaseFrames
            Signal frames object.
        max_formants: int, default=5
            The maximum number of formants that are extracted.
        lower: float, default=50.0
            Lower limit for formant frequencies (in Hz).
        upper: float, default=5450.0
            Upper limit for formant frequencies (in Hz).
        preemphasis_from: float, default=50.0
            Starting value for the preemphasis function (in Hz).
        window: str
            Window function that is applied before formant estimation.

        """
        frames = sig_frames_obj.frames

        if preemphasis_from is not None:
            pre_coef = math.exp(
                -2 * math.pi * preemphasis_from * (1 / sig_frames_obj.sr)
            )
            frames = librosa.effects.preemphasis(
                sig_frames_obj.frames, coef=pre_coef
            )
        if window is not None:
            if window == "praat_gaussian":
                win = cls._praat_gauss_window(sig_frames_obj.frame_len)
            else:
                win = get_window(
                    window, sig_frames_obj.frame_len, fftbins=False
                )
            frames = frames * win

        # Calc linear predictive coefficients
        coefs = librosa.lpc(frames, order=max_formants * 2)
        # Transform LPCs to formants
        formants = [
            cls._calc_formants(coef, sig_frames_obj.sr, lower, upper)
            for coef in coefs
        ]

        return cls(
            formants,
            sig_frames_obj.sr,
            sig_frames_obj.frame_len,
            sig_frames_obj.hop_len,
            sig_frames_obj.center,
            sig_frames_obj.pad_mode,
            max_formants,
            lower,
            upper,
            preemphasis_from,
            window,
        )

    @staticmethod
    def _calc_formants(
        coefs: np.ndarray, sr: int, lower: float = 50, upper: float = 5450
    ) -> List:
        # Function to compute complex norm
        def complex_norm(x):
            return np.sqrt(np.abs(np.real(x) ** 2) + np.abs(np.imag(x) ** 2))

        nf_pi = sr / (2 * math.pi)  # sr/2 = Nyquist freq
        # Find roots of linear coefficients
        roots = np.roots(coefs)
        # Select roots with positive imag part
        mask = np.imag(roots) > 0
        roots = roots[mask]
        # Calc angular frequency
        ang_freq = np.abs(np.arctan2(np.imag(roots), np.real(roots)))
        # Calc formant centre freq
        formant_freqs = ang_freq * nf_pi
        # Calc formant bandwidth
        formant_bws = (
            -np.log(np.apply_along_axis(complex_norm, 0, roots)) * nf_pi
        )
        # Select formants within boundaries
        in_bounds = np.logical_and(formant_freqs > lower, formant_freqs < upper)
        formants_sorted = sorted(
            list(zip(formant_freqs[in_bounds], formant_bws[in_bounds]))
        )
        return formants_sorted

    def select_formant_attr(
        self, formant_idx: int, attr_idx: int
    ) -> np.ndarray:
        return np.array(
            [
                f[formant_idx][attr_idx] if len(f) > formant_idx else np.nan
                for f in self.frames
            ]
        )


class FormantAmplitudeFrames(BaseFrames):
    """Estimate and store formant amplitudes.

    Parameters
    ----------
    frames: np.ndarray
        Formant amplitude frames of shape (num_frames, max_formants) in dB.
    lower: float
        Lower boundary for peak amplitude search interval.
    upper: float
        Upper boundary for peak amplitude search interval.
    rel_f0: bool
        Whether the amplitude is relative to the fundamental frequency amplitude.

    Notes
    -----
    Estimate the formant amplitude as the maximum amplitude of harmonics of the
    fundamental frequency within an interval ``[lower*f, upper*f]`` where `f` is the
    central frequency of the formant in each frame. If ``rel=True``, divide the amplitude by the amplitude of
    the fundamental frequency.

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
        upper: float,
        rel_f0: bool,
    ):
        self.lower = lower
        self.upper = upper
        self.rel_f0 = rel_f0
        super().__init__(frames, sr, frame_len, hop_len, center, pad_mode)

    @property
    def idx(self) -> np.ndarray:
        if self._idx is None:
            self._idx = np.arange(len(self.frames))
        return self._idx

    @classmethod
    def from_formant_harmonics_and_pitch_frames(  # pylint: disable=too-many-locals
        cls,
        formant_frames_obj: FormantFrames,
        harmonics_frames_obj: PitchHarmonicsFrames,
        pitch_frames_obj: PitchFrames,
        lower: float = 0.8,
        upper: float = 1.2,
        rel_f0: bool = True,
    ):
        """Estimate formant amplitudes from formant, pitch harmonics, and pitch frames.

        Parameters
        ----------
        formant_frames_obj: FormantFrames
            Formant frames object.
        harmonics_frames_obj: PitchHarmonicsFrames
            Pitch harmonics frames object.
        pitch_frames_obj: PitchFrames
            Pitch frames object.
        lower: float, optional, default=0.8
            Lower boundary for peak amplitude search interval.
        upper: float, optional, default=1.2
            Upper boundary for peak amplitude search interval.
        rel_f0: bool, optional, default=True
            Whether the amplitude is divided by the fundamental frequency amplitude.
        """
        amp_frames = []

        for i in range(formant_frames_obj.max_formants):
            freqs = formant_frames_obj.select_formant_attr(i, 0)
            harmonic_freqs = (
                pitch_frames_obj.frames[:, None]
                * (np.arange(harmonics_frames_obj.n_harmonics) + 1)[None, :]
            )
            f0_amp = harmonics_frames_obj.frames[:, 0]
            freqs_lower = lower * freqs
            freqs_upper = upper * freqs
            freq_in_bounds = np.logical_and(
                harmonic_freqs > freqs_lower[:, None],
                harmonic_freqs < freqs_upper[:, None],
            )
            harmonics_amp = copy(harmonics_frames_obj.frames)
            harmonics_amp[~freq_in_bounds] = np.nan
            # Set all-nan frames to nan (otherwise np.nanmax throws warning)
            harmonic_peaks = np.zeros(harmonics_amp.shape[0:1])
            harmonics_amp_all_na = np.all(np.isnan(harmonics_amp), axis=1)
            harmonic_peaks[harmonics_amp_all_na] = np.nan
            harmonic_peaks[~harmonics_amp_all_na] = np.nanmax(
                harmonics_amp[~harmonics_amp_all_na], axis=1
            )
            harmonic_peaks_db = librosa.amplitude_to_db(harmonic_peaks)

            if rel_f0:
                harmonic_peaks_db = harmonic_peaks_db - librosa.amplitude_to_db(
                    f0_amp
                )

            amp_frames.append(harmonic_peaks_db)

        return cls(
            np.array(amp_frames).T,
            formant_frames_obj.sr,
            formant_frames_obj.frame_len,
            formant_frames_obj.hop_len,
            formant_frames_obj.center,
            formant_frames_obj.pad_mode,
            lower,
            upper,
            rel_f0,
        )
