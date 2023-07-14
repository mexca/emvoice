"""Pitch-related voice features.
"""

import logging
from typing import List, Optional, Tuple

import librosa
import numpy as np
from scipy.interpolate import interp1d

from emvoice.frames import BaseFrames
from emvoice.signal import BaseSignal
from emvoice.spectral import SpecFrames


class PitchFrames(BaseFrames):
    """Estimate and store pitch frames.

    Estimate and store the voice pitch measured as the fundamental frequency F0 in Hz.

    Parameters
    ----------
    frames: numpy.ndarray
        Voice pitch frames in Hz with shape (num_frames,).
    flag: numpy.ndarray
        Boolean flags indicating which frames are voiced with shape (num_frames,).
    prob: numpy.ndarray
        Probabilities for frames being voiced with shape (num_frames,).
    lower: float
        Lower limit used for pitch estimation (in Hz).
    upper: float
        Upper limit used for pitch estimation (in Hz).
    method: str
        Method used for estimating voice pitch.

    See Also
    --------
    librosa.pyin
    librosa.yin

    """

    def __init__(
        self,
        frames: np.ndarray,
        flag: np.ndarray,
        prob: np.ndarray,
        sr: int,
        lower: float,
        upper: float,
        frame_len: int,
        hop_len: int,
        method: str,
        center: bool = True,
        pad_mode: str = "constant",
    ):
        self.logger = logging.getLogger("emvoice.frequency.PitchFrames")
        self.flag = flag
        self.prob = prob
        self.lower = lower
        self.upper = upper
        self.method = method
        super().__init__(frames, sr, frame_len, hop_len, center, pad_mode)

    @classmethod
    def from_signal(
        cls,
        sig_obj: BaseSignal,
        frame_len: int,
        hop_len: Optional[int] = None,
        center: bool = True,
        pad_mode: str = "constant",
        lower: float = 75.0,
        upper: float = 600.0,
        method: str = "pyin",
    ):
        """Estimate the voice pitch frames from a signal.

        Currently, voice pitch can only be extracted with the *pYIN* method.

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
        lower: float, default = 75.0
            Lower limit for pitch estimation (in Hz).
        upper: float, default = 600.0
            Upper limit for pitch estimation (in Hz).
        method: str, default = 'pyin'
            Method for estimating voice pitch. Only `'pyin'` is currently available.

        Raises
        ------
        NotImplementedError
            If a method other than `'pyin'` is given.

        """
        if hop_len is None:
            hop_len = frame_len // 4

        if method == "pyin":
            pitch_f0, flag, prob = librosa.pyin(
                sig_obj.sig,
                fmin=lower,
                fmax=upper,
                sr=sig_obj.sr,
                frame_length=frame_len,
                hop_length=hop_len,
                center=center,
                pad_mode=pad_mode,
            )
        else:
            raise NotImplementedError(
                'Only the "pyin" method is currently available'
            )

        return cls(
            frames=pitch_f0,
            flag=flag,
            prob=prob,
            sr=sig_obj.sr,
            lower=lower,
            upper=upper,
            frame_len=frame_len,
            hop_len=hop_len,
            method=method,
        )


class PitchPulseFrames(BaseFrames):
    """Extract and store glottal pulse frames.

    Glottal pulses are peaks in the signal corresponding to the fundamental frequency F0.

    Parameters
    ----------
    frames: list
        Pulse frames. Each frame contains a list of pulses or an empty list if no pulses are detected.
        Pulses are stored as tuples (pulse timestamp, T0, amplitude).

    Notes
    -----
    See :ref:`Algorithms section <Glottal pulses>` for details.

    """

    def __init__(
        self,
        frames: List[Tuple],
        sr: int,
        frame_len: int,
        hop_len: int,
        center: bool = True,
        pad_mode: str = "constant",
    ) -> None:
        self.logger = logging.getLogger("emvoice.frequency.PitchPulseFrames")
        super().__init__(frames, sr, frame_len, hop_len, center, pad_mode)

    @property
    def idx(self) -> np.ndarray:
        if self._idx is None:
            self._idx = np.arange(len(self.frames))
        return self._idx

    @classmethod
    def from_signal_and_pitch_frames(
        cls, sig_obj: BaseSignal, pitch_frames_obj: PitchFrames
    ):
        """Extract glottal pulse frames from a signal and voice pitch frames.

        Parameters
        ----------
        sig_obj: BaseSignal
            Signal object.
        pitch_frames_obj: PitchFrames
            Voice pitch frames object.

        """
        # Access to padded signal required so we transform it here again! Could go into separate private method perhaps
        padding = [(0, 0) for _ in sig_obj.sig.shape]
        padding[-1] = (
            pitch_frames_obj.frame_len // 2,
            pitch_frames_obj.frame_len // 2,
        )
        sig_padded = np.pad(
            sig_obj.sig, padding, mode=pitch_frames_obj.pad_mode
        )
        # Create ts for padded signal
        sig_padded_ts = librosa.samples_to_time(
            np.arange(sig_padded.shape[0]), sr=sig_obj.sr
        )

        # Frame padded signal
        sig_frames_obj = BaseFrames.from_signal(
            BaseSignal(sig_padded, sig_obj.sr),
            pitch_frames_obj.frame_len,
            pitch_frames_obj.hop_len,
            center=False,
        )

        # Frame ts of padded signal
        sig_ts_frames_obj = BaseFrames.from_signal(
            BaseSignal(sig_padded_ts, sig_obj.sr),
            pitch_frames_obj.frame_len,
            pitch_frames_obj.hop_len,
            center=False,
        )

        # Interpolate pitch F0 at padded signal ts
        interp_f0 = np.interp(
            sig_padded_ts,
            pitch_frames_obj.ts[pitch_frames_obj.flag],
            pitch_frames_obj.frames[pitch_frames_obj.flag],
        )

        # Frame F0 interpolated signal
        pitch_interp_frames_obj = BaseFrames.from_signal(
            BaseSignal(interp_f0, sig_obj.sr),
            pitch_frames_obj.frame_len,
            pitch_frames_obj.hop_len,
            center=False,
        )

        # Detect pulses in each frame; objects are passed instead of arrays bcs some attributes are needed
        pulses = [
            cls._detect_pulses_in_frame(
                i,
                sig_frames_obj,
                sig_ts_frames_obj,
                pitch_frames_obj,
                pitch_interp_frames_obj,
            )
            for i in pitch_frames_obj.idx
        ]

        return cls(
            pulses,
            pitch_frames_obj.sr,
            pitch_frames_obj.frame_len,
            pitch_frames_obj.hop_len,
            pitch_frames_obj.center,
            pitch_frames_obj.pad_mode,
        )

    @classmethod
    def _get_next_pulse(
        cls,
        sig: np.ndarray,
        ts: np.ndarray,
        t0_interp: np.ndarray,
        start: float,
        stop: float,
        left: bool = True,
        pulses: Optional[List] = None,
    ):
        # Init pulses as list if first iter of recurrence and default
        if pulses is None:
            pulses = []

        # If interval [start, stop] reaches end of frame, exit recurrence
        if (
            (left and start <= ts.min())
            or (not left and stop >= ts.max())
            or any(np.isnan((start, stop)))
        ):
            return pulses

        # Get closest ts to boundaries start, stop
        start_idx = np.argmin(np.abs(ts - start))
        stop_idx = np.argmin(np.abs(ts - stop))
        interval = sig[start_idx:stop_idx]

        # Find max peak in interval [start, stop]
        peak_idx = np.nanargmax(interval)

        # Set new mid point to idx of max peak
        new_ts_mid = ts[start_idx:stop_idx][peak_idx]

        # Add pulse to output
        new_t0_interp_mid = t0_interp[start_idx:stop_idx][peak_idx]
        pulses.append((new_ts_mid, new_t0_interp_mid, interval[peak_idx]))

        # self.logger.debug('%s - %s - %s', start, stop, pulses)

        if left:  # Move interval to left
            start = new_ts_mid - 1.25 * new_t0_interp_mid
            stop = new_ts_mid - 0.8 * new_t0_interp_mid
        else:  # Move interval to right
            stop = new_ts_mid + 1.25 * new_t0_interp_mid
            start = new_ts_mid + 0.8 * new_t0_interp_mid

        # Find next pulse in new interval
        return cls._get_next_pulse(
            sig, ts, t0_interp, start, stop, left, pulses
        )

    @classmethod
    def _detect_pulses_in_frame(
        cls,
        frame_idx: int,
        sig_frames_obj: BaseFrames,
        sig_ts_frames_obj: BaseFrames,
        pitch_obj: PitchFrames,
        pitch_interp_obj: BaseFrames,
    ) -> List[Tuple]:
        # Get period of frame
        t0_mid = 1 / pitch_obj.frames[frame_idx]
        # Get ts of frame
        ts_mid = pitch_obj.ts[frame_idx]
        # Get frame signal
        sig_frame = sig_frames_obj.frames[frame_idx, :]
        # Get ts of frame signal
        ts_sig_frame = sig_ts_frames_obj.frames[frame_idx, :]
        # Get interpolated period of frame
        t0 = 1 / pitch_interp_obj.frames[frame_idx, :]

        pulses = []

        # Return empty list if frame is unvoiced (no F0)
        if np.all(np.isnan(t0)) or np.isnan(t0_mid):
            return pulses

        # Set start interval
        start = ts_mid - t0_mid / 2
        stop = ts_mid + t0_mid / 2

        # Get pulses to the left
        cls._get_next_pulse(
            sig_frame, ts_sig_frame, t0, start, stop, True, pulses
        )

        # Get pulses to the right
        cls._get_next_pulse(
            sig_frame, ts_sig_frame, t0, start, stop, False, pulses
        )

        return list(sorted(set(pulses)))


class PitchPeriodFrames(BaseFrames):
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
    ):
        self.logger = logging.getLogger("emvoice.frequency.PitchPeriodFrames")
        self.lower = lower
        self.upper = upper
        super().__init__(frames, sr, frame_len, hop_len, center, pad_mode)

    @staticmethod
    def _calc_period_length(
        pulses: List[Tuple], lower: float, upper: float
    ) -> Tuple[List, np.ndarray]:
        # Calc period length as first order diff of pulse ts
        periods = np.diff(np.array([puls[0] for puls in pulses]))

        # Filter out too short and long periods
        mask = np.logical_and(periods > lower, periods < upper)

        # Split periods according to mask and remove masked periods
        periods = np.array_split(periods[mask], np.where(~mask)[0])

        return periods, mask

    @staticmethod
    def _check_ratio(x_arr: np.ndarray, threshold: float) -> np.ndarray:
        valid = np.logical_and(np.isfinite(x_arr[1:]), x_arr[1:] > 0)
        valid[valid] = x_arr[:-1][valid] / x_arr[1:][valid] < threshold
        return valid


class PitchHarmonicsFrames(BaseFrames):
    """Estimate and store voice pitch harmonics.

    Compute the energy of the signal at harmonics (`nF0` for any integer n) of
    the fundamental frequency.

    Parameters
    ----------
    frames: numpy.ndarray
        Harmonics frames with the shape (num_frames, n_harmonics)
    n_harmonics: int, default=100
        Number of estimated harmonics.

    See Also
    --------
    librosa.f0_harmonics

    """

    def __init__(
        self,
        frames: np.ndarray,
        sr: int,
        frame_len: int,
        hop_len: int,
        center: bool = True,
        pad_mode: str = "constant",
        n_harmonics: int = 100,
    ):
        self.logger = logging.getLogger(
            "emvoice.frequency.PitchHarmonicsFrames"
        )
        self.n_harmonics = n_harmonics
        super().__init__(frames, sr, frame_len, hop_len, center, pad_mode)

    @classmethod
    def from_spec_and_pitch_frames(
        cls,
        spec_frames_obj: SpecFrames,
        pitch_frames_obj: PitchFrames,
        n_harmonics: int = 100,
    ):
        """Estimate voice pitch harmonics from spectrogram frames and voice pitch frames.

        Parameters
        ----------
        spec_frames_obj: SpecFrames
            Spectrogram frames object.
        pitch_frames_obj: PitchFrames
            Pitch frames object.
        n_harmonics: int, default=100
            Number of harmonics to estimate.

        """

        # harmonics = librosa.f0_harmonics(
        #     np.abs(spec_frames_obj.frames),
        #     freqs=freqs,
        #     f0=pitch_frames_obj.frames,
        #     harmonics=np.arange(n_harmonics) + 1,  # Shift one up
        #     axis=-1,
        # )

        harmonics = cls._calc_f0_harmonics(
            spec_frames_obj.frames,
            spec_frames_obj.freqs,
            pitch_frames_obj.frames,
            n_harmonics,
        )

        return cls(
            harmonics,
            spec_frames_obj.sr,
            spec_frames_obj.frame_len,
            spec_frames_obj.hop_len,
            spec_frames_obj.center,
            spec_frames_obj.pad_mode,
            n_harmonics,
        )

    @staticmethod
    def _calc_f0_harmonics(
        spec_frames: np.ndarray,
        freqs: np.ndarray,
        f0_frames: np.ndarray,
        n_harmonics: int,
    ) -> np.ndarray:
        # Adapted from librosa.f0_harmonics, see:
        # https://librosa.org/doc/latest/generated/librosa.f0_harmonics.html#librosa.f0_harmonics
        is_valid = np.isfinite(freqs)

        def mag_interp_fun(spec_frames, f0_harmonic_freqs):
            interp = interp1d(
                freqs[is_valid],
                spec_frames[is_valid],
                axis=0,
                copy=False,
                assume_sorted=False,
                bounds_error=False,
                fill_value=0,
            )
            return interp(f0_harmonic_freqs)

        xfunc = np.vectorize(mag_interp_fun, signature="(f),(h)->(h)")
        harmonics_frames = xfunc(
            np.abs(spec_frames),
            np.multiply.outer(
                f0_frames, np.arange(n_harmonics) + 1
            ),  # Shift one up
        )

        return harmonics_frames


class JitterFrames(PitchPeriodFrames):
    """Extract and store voice jitter frames.

    Parameters
    ----------
    frames: numpy.ndarray
        Voice jitter frames of shape (num_frames,).
    rel: bool
        Whether the voice jitter is relative to the average period length.
    lower: float
        Lower limit for periods between glottal pulses.
    upper: float
        Upper limit for periods between glottal pulses.
    max_period_ratio: float
        Maximum ratio between consecutive periods used for jitter extraction.

    Notes
    -----
    Compute jitter as the average absolute difference between consecutive fundamental periods with a ratio
    below `max_period_ratio` for each frame. If ``rel=True``, jitter is divided by the average fundamental period
    of each frame. Fundamental periods are calculated as the first-order temporal difference between consecutive
    glottal pulses.

    """

    def __init__(
        self,
        frames: np.ndarray,
        sr: int,
        frame_len: int,
        hop_len: int,
        center: bool,
        pad_mode: str,
        rel: bool,
        lower: float,
        upper: float,
        max_period_ratio: float,
    ):
        self.logger = logging.getLogger("emvoice.frequency.JitterFrames")
        self.rel = rel
        self.max_period_ratio = max_period_ratio
        super().__init__(
            frames, sr, frame_len, hop_len, center, pad_mode, lower, upper
        )

    @classmethod
    def from_pitch_pulse_frames(
        cls,
        pitch_pulse_frames_obj: PitchPulseFrames,
        rel: bool = True,
        lower: float = 0.0001,
        upper: float = 0.02,
        max_period_ratio: float = 1.3,
    ):
        """Extract voice jitter frames from glottal pulse frames.

        Parameters
        ----------
        pitch_pulse_frames_obj: PitchPulseFrames
            Glottal pulse frames object.
        rel: bool, optional, default=True
            Divide jitter by the average pitch period.
        lower: float, optional, default=0.0001
            Lower limit for periods between glottal pulses.
        upper: float, optional, default=0.02
            Upper limit for periods between glottal pulses.
        max_period_ratio: float, optional, default=1.3
            Maximum ratio between consecutive periods for jitter extraction.
        """
        jitter_frames = np.array(
            [
                cls._calc_jitter_frame(
                    pulses, rel, lower, upper, max_period_ratio
                )
                for pulses in pitch_pulse_frames_obj.frames
            ]
        )

        return cls(
            jitter_frames,
            pitch_pulse_frames_obj.sr,
            pitch_pulse_frames_obj.frame_len,
            pitch_pulse_frames_obj.hop_len,
            pitch_pulse_frames_obj.center,
            pitch_pulse_frames_obj.pad_mode,
            rel,
            lower,
            upper,
            max_period_ratio,
        )

    @classmethod
    def _calc_jitter_frame(
        cls,
        pulses: List[Tuple],
        rel: bool,
        lower: float,
        upper: float,
        max_period_ratio: float,
    ):
        if len(pulses) == 0:
            return np.nan

        # Calc period length as first order diff of pulse ts
        periods, _ = cls._calc_period_length(pulses, lower, upper)

        if len(periods) == 0 or all(len(period) <= 1 for period in periods):
            return np.nan

        # Calc avg of first order diff in period length
        # only consider period pairs where ratio is < max_period_ratio
        period_diff = [
            np.abs(np.diff(period)[cls._check_ratio(period, max_period_ratio)])
            for period in periods
            if len(period) > 1
        ]

        if len(period_diff) == 0 or all(
            len(period) == 0 for period in period_diff
        ):
            return np.nan

        avg_period_diff = np.nanmean(
            np.array([np.mean(period) for period in period_diff])
        )

        if rel:  # Relative to mean period length
            avg_period_len = np.nanmean(
                np.array(
                    [np.mean(period) for period in periods if len(period) > 1]
                )
            )
            return avg_period_diff / avg_period_len

        return avg_period_diff


class ShimmerFrames(PitchPeriodFrames):
    """Extract and store voice shimmer frames.

    Parameters
    ----------
    frames: numpy.ndarray
        Voice shimmer frames of shape (num_frames,).
    rel: bool
        Whether the voice shimmer is relative to the average period length.
    lower: float
        Lower limit for periods between glottal pulses.
    upper: float
        Upper limit for periods between glottal pulses.
    max_period_ratio: float
        Maximum ratio between consecutive periods used for shimmer extraction.
    max_amp_factor: float
        Maximum ratio between consecutive amplitudes used for shimmer extraction.

    Notes
    -----
    Compute shimmer as the average absolute difference between consecutive pitch amplitudes with a
    fundamental period ratio below `max_period_ratio` and amplitude ratio below `max_amp_factor`
    for each frame. If ``rel=True``, shimmer is divided by the average amplitude
    of each frame. Fundamental periods are calculated as the first-order temporal difference
    between consecutive glottal pulses. Amplitudes are signal amplitudes at the glottal pulses.
    """

    def __init__(
        self,
        frames: List[Tuple],
        sr: int,
        frame_len: int,
        hop_len: int,
        center: bool,
        pad_mode: str,
        rel: bool,
        lower: float,
        upper: float,
        max_period_ratio: float,
        max_amp_factor: float,
    ):
        self.logger = logging.getLogger("emvoice.frequency.ShimmerFrames")
        self.rel = rel
        self.max_period_ratio = max_period_ratio
        self.max_amp_factor = max_amp_factor
        super().__init__(
            frames, sr, frame_len, hop_len, center, pad_mode, lower, upper
        )

    @classmethod
    def from_pitch_pulse_frames(
        cls,
        pitch_pulse_frames_obj: PitchPulseFrames,
        rel: bool = True,
        lower: float = 0.0001,
        upper: float = 0.02,
        max_period_ratio: float = 1.3,
        max_amp_factor: float = 1.6,
    ):
        """Extract voice shimmer frames from glottal pulse frames.

        Parameters
        ----------
        pitch_pulse_frames_obj: PitchPulseFrames
            Glottal pulse frames object.
        rel: bool, optional, default=True
            Divide shimmer by the average pulse amplitude.
        lower: float, optional, default=0.0001
            Lower limit for periods between glottal pulses.
        upper: float, optional, default=0.02
            Upper limit for periods between glottal pulses.
        max_period_ratio: float, optional, default=1.3
            Maximum ratio between consecutive periods for shimmer extraction.
        max_amp_factor: float, optional, default=1.6
            Maximum ratio between consecutive amplitudes used for shimmer extraction.
        """
        shimmer_frames = np.array(
            [
                cls._calc_shimmer_frame(
                    pulses, rel, lower, upper, max_period_ratio, max_amp_factor
                )
                for pulses in pitch_pulse_frames_obj.frames
            ]
        )

        return cls(
            shimmer_frames,
            pitch_pulse_frames_obj.sr,
            pitch_pulse_frames_obj.frame_len,
            pitch_pulse_frames_obj.hop_len,
            pitch_pulse_frames_obj.center,
            pitch_pulse_frames_obj.pad_mode,
            rel,
            lower,
            upper,
            max_period_ratio,
            max_amp_factor,
        )

    @classmethod
    def _calc_shimmer_frame(
        cls,
        pulses: List[Tuple],
        rel: bool,
        lower: float,
        upper: float,
        max_period_ratio: float,
        max_amp_factor: float,
    ) -> float:
        if len(pulses) == 0:
            return np.nan

        # Calc period length as first order diff of pulse ts
        periods, mask = cls._calc_period_length(pulses, lower, upper)
        amps = cls._get_amplitude(pulses, mask)

        if (
            len(periods) == 0
            or len(amps) == 0
            or all(len(period) <= 1 for period in periods)
        ):
            return np.nan

        # Calc avg of first order diff in amplitude
        # only consider period pairs where period ratio is < max_period_ratio and
        # where amplitude ratio is < max_amp_factor
        amp_diff = [
            np.abs(
                np.diff(amp)[
                    np.logical_and(
                        cls._check_ratio(period, max_period_ratio),
                        cls._check_ratio(amp, max_amp_factor),
                    )
                ]
            )
            for amp, period in zip(amps, periods)
            if len(period) > 1 and len(amp) > 1
        ]

        if len(amp_diff) == 0 or all(len(amp) == 0 for amp in amp_diff):
            return np.nan

        avg_amp_diff = np.nanmean(np.array([np.mean(amp) for amp in amp_diff]))

        if rel:  # Relative to mean amplitude
            avg_amp = np.nanmean(
                np.array([np.mean(amp) for amp in amps if len(amp) > 1])
            )
            return avg_amp_diff / avg_amp

        return avg_amp_diff

    @staticmethod
    def _get_amplitude(pulses: List[Tuple], mask: np.ndarray) -> List:
        # Get amplitudes
        amps = np.array([puls[2] for puls in pulses])[
            1:
        ]  # Skip first amplitude to align with periods

        # Split periods according to mask and remove masked periods
        amps = np.array_split(amps[mask], np.where(~mask)[0])

        return amps
