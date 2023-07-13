"""Store audio and voice feature frames."""

import logging
from typing import Optional

import librosa
import numpy as np

from emvoice.signal import BaseSignal


class BaseFrames:
    """Create and store signal frames.

    A frame is an (overlapping, padded) slice of a signal for which higher-order
    features can be computed.

    Parameters
    ----------
    frames: numpy.ndarray
        Signal frames. The first dimension should be the number of frames.
    sr: int
        Sampling rate.
    frame_len: int
        Number of samples per frame.
    hop_len: int
        Number of samples between frame starting points.
    center: bool, default=True
        Whether the signal has been centered and padded before framing.
    pad_mode: str, default='constant'
        How the signal has been padded before framing. See :func:`numpy.pad`.
        Uses the default value 0 for `'constant'` padding.

    See Also
    --------
    librosa.util.frame

    """

    _ts: Optional[np.ndarray] = None
    _idx: Optional[np.ndarray] = None

    def __init__(
        self,
        frames: np.ndarray,
        sr: int,
        frame_len: int,
        hop_len: int,
        center: bool = True,
        pad_mode: str = "constant",
    ) -> None:
        self.logger = logging.getLogger("emvoice.frames.BaseFrames")
        self.frames = frames
        self.sr = sr
        self.frame_len = frame_len
        self.hop_len = hop_len
        self.center = center
        self.pad_mode = pad_mode

    @property
    def idx(self) -> np.ndarray:
        """Frame indices (read-only)."""
        if self._idx is None:
            self._idx = np.arange(self.frames.shape[0])
        return self._idx

    @property
    def ts(self) -> np.ndarray:
        """Frame timestamps (read-only)."""
        if self._ts is None:
            self._ts = librosa.frames_to_time(
                self.idx, sr=self.sr, hop_length=self.hop_len
            )
        return self._ts

    @classmethod
    def from_signal(
        cls,
        sig_obj: BaseSignal,
        frame_len: int,
        hop_len: Optional[int] = None,
        center: bool = True,
        pad_mode: str = "constant",
    ):
        """Create frames from a signal.

        Parameters
        ----------
        sig_obj: BaseSignal
            Signal object.
        frame_len: int
            Number of samples per frame.
        hop_len: int, optional, default=None
            Number of samples between frame starting points. If `None`, uses `frame_len // 4`.
        center: bool, default=True
            Whether to center the frames and apply padding.
        pad_mode: str, default='constant'
            How the signal is padded before framing. See :func:`numpy.pad`.
            Uses the default value 0 for `'constant'` padding. Ignored if `center=False`.
        """
        if hop_len is None:
            hop_len = frame_len // 4
        sig = sig_obj.sig
        # Pad signal if centering
        if center:
            padding = [(0, 0) for _ in sig_obj.sig.shape]
            padding[-1] = (frame_len // 2, frame_len // 2)
            sig = np.pad(sig, padding, mode=pad_mode)

        frames = librosa.util.frame(
            sig, frame_length=frame_len, hop_length=hop_len, axis=0
        )

        return cls(frames, sig_obj.sr, frame_len, hop_len, center, pad_mode)
