"""Load and store audio signals."""

import logging
import math
from typing import Optional

import librosa
import numpy as np


class BaseSignal:
    """Store a signal.

    Parameters
    ----------
    sig: numpy.ndarray
        Signal.
    sr: int
        Sampling rate.
    """

    _ts: Optional[np.ndarray] = None
    _idx: Optional[np.ndarray] = None

    def __init__(self, sig: np.ndarray, sr: int) -> None:
        self.logger = logging.getLogger("emvoice.signal.BaseSignal")
        self.sig = sig
        self.sr = sr

    @property
    def idx(self) -> np.ndarray:
        """Sample indices (read-only)."""
        if self._idx is None:
            self._idx = np.arange(self.sig.shape[0])
        return self._idx

    @property
    def ts(self) -> np.ndarray:
        """Sample timestamps (read-only)."""
        if self._ts is None:
            self._ts = librosa.samples_to_time(self.idx, sr=self.sr)
        return self._ts


class AudioSignal(BaseSignal):
    """Load and store an audio signal.

    Parameters
    ----------
    sig: numpy.ndarray
        Audio signal.
    sr: int
        Sampling rate.
    mono: bool, default=True
        Whether the signal has been converted to mono or not.
    filename: str, optional
        Name of the audio file associated with the signal.
    """

    def __init__(
        self,
        sig: np.ndarray,
        sr: int,
        mono: bool = True,
        filename: Optional[str] = None,
    ) -> None:
        self.logger = logging.getLogger("emvoice.signal.AudioSignal")
        self.filename = filename
        self.mono = mono
        super().__init__(sig, sr)

    @classmethod
    def from_file(
        cls, filename: str, sr: Optional[float] = None, mono: bool = True
    ):
        """Load a signal from an audio file.

        Parameters
        ----------
        filename: str
            Name of the audio file.
            File types must be supported by ``soundfile`` or ``audiofile``.
            See :func:`librosa.load`.
        sr: float, optional, default=None
            Sampling rate. If `None`, is detected from the file, otherwise the signal is resampled.
        mono: bool, default=True
            Whether to convert the signal to mono.
        """
        sig, nat_sr = librosa.load(path=filename, sr=sr, mono=mono)
        return cls(sig, nat_sr, mono, filename)


class FormantAudioSignal(AudioSignal):
    def __init__(
        self,
        sig: np.ndarray,
        sr: int,
        mono: bool,
        filename: Optional[str],
        preemphasis_from: Optional[float],
    ):
        self.preemphasis_from = preemphasis_from
        super().__init__(sig, sr, mono, filename)

    @staticmethod
    def _calc_preemphasis_coef(preemphasis_from: float, sr: float) -> float:
        return math.exp(-2 * math.pi * preemphasis_from * (1 / sr))

    @classmethod
    def from_file(
        cls,
        filename: str,
        sr: Optional[float] = None,
        mono: bool = True,
        preemphasis_from: Optional[float] = 50.0,
    ):
        """Load a signal from an audio file.

        Parameters
        ----------
        filename: str
            Name of the audio file.
            File types must be supported by ``soundfile`` or ``audiofile``.
            See :func:`librosa.load`.
        sr: float, optional, default=None
            Sampling rate. If `None`, is detected from the file, otherwise the signal is resampled.
        mono: bool, default=True
            Whether to convert the signal to mono.
        preemphasis_from: float, optional, default=50.0
            Pre-emphasize the signal from this value onwards (in Hz).
        """
        audio_sig_obj = super().from_file(filename, sr, mono)
        return cls.from_audio_signal(audio_sig_obj, preemphasis_from)

    @classmethod
    def from_audio_signal(
        cls,
        audio_sig_obj: AudioSignal,
        preemphasis_from: Optional[float] = 50.0,
    ):
        sig = audio_sig_obj.sig

        if preemphasis_from is not None:
            pre_coef = cls._calc_preemphasis_coef(
                preemphasis_from, audio_sig_obj.sr
            )
            sig = librosa.effects.preemphasis(sig, coef=pre_coef)

        return cls(
            sig,
            audio_sig_obj.sr,
            audio_sig_obj.mono,
            audio_sig_obj.filename,
            preemphasis_from,
        )

    @staticmethod
    def _preemphasize(sig: np.ndarray, sr: float, preemphasis_from: float):
        pre_coef = math.exp(-2 * math.pi * preemphasis_from * (1 / sr))
        return librosa.effects.preemphasis(sig, coef=pre_coef)
