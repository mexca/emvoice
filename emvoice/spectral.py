"""Spectral voice features."""

import logging
from typing import Optional, Tuple, Union

import librosa
import numpy as np
from sklearn.linear_model import LinearRegression

from emvoice.frames import BaseFrames
from emvoice.signal import BaseSignal


class SpecFrames(BaseFrames):
    """Create and store spectrogram frames.

    Computes a spectrogram of a signal using the short-time Fourier transform (STFT).

    Parameters
    ----------
    frames: numpy.ndarray
        Spectrogram frames.
    window: str
        The window that was applied before the STFT.

    Notes
    -----
    Frames contain complex arrays `x` where ``np.abs(x)`` is the magnitude and
    ``np.angle(x)`` is the phase of the signal for different frequency bins.

    See Also
    --------
    librosa.stft

    """

    def __init__(
        self,
        frames: np.ndarray,
        sr: int,
        window: str,
        frame_len: int,
        hop_len: int,
        center: bool = True,
        pad_mode: str = "constant",
    ) -> None:
        self.logger = logging.getLogger("emvoice.frequency.SpecFrames")
        self.window = window
        self._freqs = None
        super().__init__(frames, sr, frame_len, hop_len, center, pad_mode)

    @property
    def freqs(self):
        if self._freqs is None:
            self._freqs = librosa.fft_frequencies(
                sr=self.sr, n_fft=self.frame_len
            )
        return self._freqs

    @classmethod
    def from_signal(
        cls,
        sig_obj: BaseSignal,
        frame_len: int,
        hop_len: Optional[int] = None,
        center: bool = True,
        pad_mode: str = "constant",
        window: Union[str, float, Tuple] = "hann",
    ):
        """Transform a signal into spectrogram frames.

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
        window: str
            The window that is applied before the STFT.
        """
        spec_frames = librosa.stft(
            sig_obj.sig,
            n_fft=frame_len,
            hop_length=hop_len,
            window=window,
            center=center,
            pad_mode=pad_mode,
        )
        return cls(
            np.swapaxes(spec_frames, 0, 1),
            sig_obj.sr,
            window,
            frame_len,
            hop_len,
            center,
            pad_mode,
        )


class MelSpecFrames(SpecFrames):
    """Calculate and store Mel spectrograms.

    Parameters
    ----------
    frames: numpy.ndarray
        Spectrogram frames on the Mel power scale with shape (num_frames, n_mels).
    n_mels: int
        Number of Mel filters.
    lower: float
        Lower frequency boundary in Hz.
    upper: float
        Upper frequency boundary in Hz.

    See Also
    --------
    librosa.feature.melspectrogram

    """

    def __init__(
        self,
        frames: np.ndarray,
        sr: int,
        window: str,
        frame_len: int,
        hop_len: int,
        center: bool,
        pad_mode: str,
        n_mels: int,
        lower: float,
        upper: float,
    ):
        self.logger = logging.getLogger("emvoice.frequency.MelSpecFrames")
        self.n_mels = n_mels
        self.lower = lower
        self.upper = upper
        super().__init__(
            frames, sr, window, frame_len, hop_len, center, pad_mode
        )

    @classmethod
    def from_spec_frames(
        cls,
        spec_frames_obj: SpecFrames,
        n_mels: int = 26,
        lower: float = 20.0,
        upper: float = 8000.0,
    ):
        """Calculate Mel spectrograms from spectrogram frames.

        spec_frames_obj: SpecFrames
            Spectrogram frames object.
        n_mels: int, default=26
            Number of Mel filters.
        lower: float, default=20.0
            Lower frequency boundary in Hz.
        upper: float, default=8000.0
            Upper frequency boundary in Hz.

        """
        mel_spec_frames = librosa.feature.melspectrogram(
            S=np.abs(spec_frames_obj.frames.T) ** 2,  # requires power spectrum
            sr=spec_frames_obj.sr,
            n_fft=spec_frames_obj.frame_len,
            hop_length=spec_frames_obj.hop_len,
            window=spec_frames_obj.window,
            center=spec_frames_obj.center,
            pad_mode=spec_frames_obj.pad_mode,
            n_mels=n_mels,
            fmin=lower,
            fmax=upper,
        )

        return cls(
            mel_spec_frames.T,  # outputs power spectrum
            spec_frames_obj.sr,
            spec_frames_obj.window,
            spec_frames_obj.frame_len,
            spec_frames_obj.hop_len,
            spec_frames_obj.center,
            spec_frames_obj.pad_mode,
            n_mels,
            lower,
            upper,
        )


class MfccFrames(MelSpecFrames):
    """Estimate and store Mel frequency cepstral coefficients (MFCCs).

    Parameters
    ----------
    frames: numpy.ndarray
        MFCC frames with shape (num_frames, n_mfcc).
    n_mfcc: int
        Number of coeffcients that were estimated per frame.
    lifter: float
        Cepstral liftering coefficient. Must be >= 0. If zero, no liftering is applied.


    """

    def __init__(
        self,
        frames: np.ndarray,
        sr: int,
        window: str,
        frame_len: int,
        hop_len: int,
        center: bool,
        pad_mode: str,
        n_mels: int,
        lower: float,
        upper: float,
        n_mfcc: int,
        lifter: float,
    ):
        self.logger = logging.getLogger("emvoice.frequency.MfccFrames")
        self.n_mfcc = n_mfcc
        self.lifter = lifter
        super().__init__(
            frames,
            sr,
            window,
            frame_len,
            hop_len,
            center,
            pad_mode,
            n_mels,
            lower,
            upper,
        )

    @classmethod
    def from_mel_spec_frames(
        cls,
        mel_spec_frames_obj: MelSpecFrames,
        n_mfcc: int = 4,
        lifter: float = 22.0,
    ):
        """Estimate MFCCs from Mel spectogram frames.

        Parameters
        ----------
        mel_spec_frames_obj: MelSpecFrames
            Mel spectrogram frames object.
        n_mfcc: int, default=4
            Number of coeffcients that were estimated per frame.
        lifter: float, default=22.0
            Cepstral liftering coefficient. Must be >= 0. If zero, no liftering is applied.

        See Also
        --------
        librosa.feature.mfcc

        """
        mfcc_frames = librosa.feature.mfcc(
            S=librosa.power_to_db(
                mel_spec_frames_obj.frames.T
            ),  # dB on power spectrum
            sr=mel_spec_frames_obj.sr,
            n_mfcc=n_mfcc,
            lifter=lifter,
        )

        return cls(
            mfcc_frames.T,
            mel_spec_frames_obj.sr,
            mel_spec_frames_obj.window,
            mel_spec_frames_obj.frame_len,
            mel_spec_frames_obj.hop_len,
            mel_spec_frames_obj.center,
            mel_spec_frames_obj.pad_mode,
            mel_spec_frames_obj.n_mels,
            mel_spec_frames_obj.lower,
            mel_spec_frames_obj.upper,
            n_mfcc,
            lifter,
        )


class AlphaRatioFrames(BaseFrames):
    """Calculate and store spectogram alpha ratios.

    Parameters
    ----------
    frames: numpy.ndarray
        Alpha ratio frames in dB with shape (num_frames,).
    lower_band: tuple
        Boundaries of the lower frequency band (start, end) in Hz.
    upper_band: tuple
        Boundaries of the upper frequency band (start, end) in Hz.

    Notes
    -----
    Calculate the alpha ratio by dividing the energy (sum of magnitude) in the lower frequency band
    by the energy in the upper frequency band. The ratio is then converted to dB.

    """

    def __init__(
        self,
        frames: np.ndarray,
        sr: int,
        frame_len: int,
        hop_len: int,
        center: bool,
        pad_mode: str,
        lower_band: Tuple[float],
        upper_band: Tuple[float],
    ):
        self.logger = logging.getLogger("emvoice.frequency.AlphaRatioFrames")
        self.lower_band = lower_band
        self.upper_band = upper_band
        super().__init__(frames, sr, frame_len, hop_len, center, pad_mode)

    @classmethod
    def from_spec_frames(
        cls,
        spec_frames_obj: SpecFrames,
        lower_band: Tuple = (50.0, 1000.0),
        upper_band: Tuple = (1000.0, 5000.0),
    ):
        """Calculate the alpha ratio from spectrogram frames.

        Parameters
        ----------
        spec_frames_obj: SpecFrames
            Spectrogram frames object.
        lower_band: tuple, default=(50.0, 1000.0)
            Boundaries of the lower frequency band (start, end) in Hz.
        upper_band: tuple, default=(1000.0, 5000.0)
            Boundaries of the upper frequency band (start, end) in Hz.

        """
        lower_band_bins = np.logical_and(
            spec_frames_obj.freqs > lower_band[0],
            spec_frames_obj.freqs <= lower_band[1],
        )
        lower_band_energy = np.nansum(
            np.abs(spec_frames_obj.frames[:, lower_band_bins]), axis=1
        )
        upper_band_bins = np.logical_and(
            spec_frames_obj.freqs > upper_band[0],
            spec_frames_obj.freqs <= upper_band[1],
        )
        upper_band_energy = np.nansum(
            np.abs(spec_frames_obj.frames[:, upper_band_bins]), axis=1
        )
        alpha_ratio_frames = np.zeros(lower_band_energy.shape)

        upper_band_energy_is_valid = np.logical_and(
            np.isfinite(upper_band_energy), upper_band_energy != 0
        )

        alpha_ratio_frames[~upper_band_energy_is_valid] = np.nan
        alpha_ratio_frames[upper_band_energy_is_valid] = (
            lower_band_energy[upper_band_energy_is_valid]
            / upper_band_energy[upper_band_energy_is_valid]
        )

        alpha_ratio_frames_db = 20.0 * np.log10(alpha_ratio_frames)

        return cls(
            alpha_ratio_frames_db,
            spec_frames_obj.sr,
            spec_frames_obj.frame_len,
            spec_frames_obj.hop_len,
            spec_frames_obj.center,
            spec_frames_obj.pad_mode,
            lower_band,
            upper_band,
        )


class HammarIndexFrames(BaseFrames):
    """Calculate and store the spectogram Hammarberg index.

    Parameters
    ----------
    frames: numpy.ndarray
        Hammarberg index frames in dB with shape (num_frames,).
    pivot_point: float
        Point separating the lower and upper frequency regions in Hz.
    upper: float
        Upper limit for the upper frequency region in Hz.

    Notes
    -----
    Calculate the Hammarberg index by dividing the peak magnitude in the spectrogram region below `pivot_point`
    by the peak magnitude in region between `pivot_point` and `upper`. The ratio is then converted to dB.

    """

    def __init__(
        self,
        frames: np.ndarray,
        sr: int,
        frame_len: int,
        hop_len: int,
        center: bool,
        pad_mode: str,
        pivot_point: float,
        upper: float,
    ):
        self.logger = logging.getLogger("emvoice.frequency.HammarIndexFrames")
        self.pivot_point = pivot_point
        self.upper = upper
        super().__init__(frames, sr, frame_len, hop_len, center, pad_mode)

    @classmethod
    def from_spec_frames(
        cls,
        spec_frames_obj: SpecFrames,
        pivot_point: float = 2000.0,
        upper: float = 5000.0,
    ):
        """Calculate the Hammarberg index from spectrogram frames.

        Parameters
        ----------
        spec_frames_obj: SpecFrames
            Spectrogram frames object.
        pivot_point: float, default=2000.0
            Point separating the lower and upper frequency regions in Hz.
        upper: float, default=5000.0
            Upper limit for the upper frequency region in Hz.

        """
        lower_band = np.abs(
            spec_frames_obj.frames[:, spec_frames_obj.freqs <= pivot_point]
        )
        upper_band_freqs = np.logical_and(
            spec_frames_obj.freqs > pivot_point, spec_frames_obj.freqs <= upper
        )
        upper_band = np.abs(spec_frames_obj.frames[:, upper_band_freqs])

        hammar_index_frames = np.zeros(lower_band.shape[0])

        upper_band_is_valid = np.logical_and(
            np.any(np.isfinite(upper_band), axis=1),
            np.all(upper_band > 0, axis=1),
        )

        hammar_index_frames[~upper_band_is_valid] = np.nan
        hammar_index_frames[upper_band_is_valid] = np.nanmax(
            lower_band[upper_band_is_valid, :], axis=1
        ) / np.nanmax(upper_band[upper_band_is_valid, :], axis=1)

        hammar_index_frames_db = librosa.amplitude_to_db(hammar_index_frames)

        return cls(
            hammar_index_frames_db,
            spec_frames_obj.sr,
            spec_frames_obj.frame_len,
            spec_frames_obj.hop_len,
            spec_frames_obj.center,
            spec_frames_obj.pad_mode,
            pivot_point,
            upper,
        )


class SpectralSlopeFrames(BaseFrames):
    """Estimate and store spectral slopes.

    Parameters
    ----------
    frames: numpy.ndarray
        Spectral slope frames with shape (num_frames, num_bands).
    bands: tuple
        Frequency bands in Hz for which slopes were estimated.

    Notes
    -----
    Estimate spectral slopes by fitting linear models to frequency bands predicting power in dB from frequency in Hz.
    Fits separate models for each frame and band.

    """

    def __init__(
        self,
        frames: np.ndarray,
        sr: int,
        frame_len: int,
        hop_len: int,
        center: bool,
        pad_mode: str,
        bands: Tuple[Tuple[float]],
    ):
        self.logger = logging.getLogger("emvoice.frequency.HammarIndexFrames")
        self.bands = bands
        super().__init__(frames, sr, frame_len, hop_len, center, pad_mode)

    @classmethod
    def from_spec_frames(
        cls,
        spec_frames_obj: SpecFrames,
        bands: Tuple[Tuple[float]] = ((0.0, 500.0), (500.0, 1500.0)),
    ):
        """Estimate spectral slopes from spectrogram frames.

        Parameters
        ----------
        spec_frames_obj: SpecFrames
            Spectrogram frames object.
        bands: tuple, default=((0.0, 500.0), (500.0, 1500.0))
            Frequency bands in Hz for which slopes are estimated.

        """
        spectral_slopes = np.zeros(
            shape=(spec_frames_obj.idx.shape[0], len(bands))
        )

        for i, band in enumerate(bands):
            band_freqs_mask = np.logical_and(
                spec_frames_obj.freqs > band[0],
                spec_frames_obj.freqs <= band[1],
            )
            band_power = np.abs(spec_frames_obj.frames[:, band_freqs_mask])
            band_freqs = spec_frames_obj.freqs[band_freqs_mask]
            spectral_slopes[:, i] = np.apply_along_axis(
                cls._calc_spectral_slope, 1, band_power, band_freqs=band_freqs
            ).squeeze()

        return cls(
            spectral_slopes,
            spec_frames_obj.sr,
            spec_frames_obj.frame_len,
            spec_frames_obj.hop_len,
            spec_frames_obj.center,
            spec_frames_obj.pad_mode,
            bands,
        )

    @staticmethod
    def _calc_spectral_slope(
        band_power: np.ndarray, band_freqs: np.ndarray
    ) -> np.ndarray:
        band_power_is_valid = np.logical_and(
            np.isfinite(band_power), band_power > 0
        )

        if np.all(~band_power_is_valid):
            return np.nan

        band_freqs_finite = band_freqs[band_power_is_valid]
        band_power_finite_db = librosa.amplitude_to_db(
            band_power[band_power_is_valid]
        )

        linear_model = LinearRegression()
        linear_model.fit(band_freqs_finite.reshape(-1, 1), band_power_finite_db)
        return linear_model.coef_


class SpectralFluxFrames(SpecFrames):
    """Calculate and store spectral flux.

    Parameters
    ----------
    frames: numpy.ndarray
        Spectral flux frames with shape (num_frames-1,).
    lower: float
        Lower limit for frequency bins.
    upper: float
        Upper limit for frequency bins

    Notes
    -----
    Compute the spectral flux as:

    1. Compute the normalized magnitudes of the frame spectra by dividing the magnitude
       at each frequency bin by the sum of all frequency bins.
    2. Compute the first-order difference of normalized magnitudes for each frequency bin within [`lower`, `upper`) across frames.
    3. Sum up the squared differences for each frame.

    Due to the first-order difference, the object has a frame less than the
    spectrogram from which it has been computed.


    """

    def __init__(
        self,
        frames: np.ndarray,
        sr: int,
        window: str,
        frame_len: int,
        hop_len: int,
        center: bool,
        pad_mode: str,
        lower: float,
        upper: float,
    ) -> None:
        self.logger = logging.getLogger("emvoice.frequency.SpectralFluxFrames")
        self.lower = lower
        self.upper = upper
        super().__init__(
            frames, sr, window, frame_len, hop_len, center, pad_mode
        )

    @classmethod
    def from_spec_frames(
        cls,
        spec_frames_obj: SpecFrames,
        lower: float = 0.0,
        upper: float = 5000.0,
    ):
        """Calculate the spectral flux from spectrogram frames.

        Parameters
        ----------
        spec_frames_obj: SpecFrames
            Spectrogram frames object.
        lower: float, default=0.0
            Lower limit for frequency bins.
        upper: float, default=5000.0
            Upper limit for frequency bins

        """
        spec_freq_mask = np.logical_and(
            spec_frames_obj.freqs >= lower, spec_frames_obj.freqs < upper
        )
        spec_mag = np.abs(spec_frames_obj.frames)
        spec_norm = np.sum(spec_mag, axis=1)
        spec_diff = np.diff(
            spec_mag[:, spec_freq_mask] / spec_norm[:, None], axis=0
        )
        spec_flux_frames = np.sum(spec_diff**2, axis=1)

        return cls(
            spec_flux_frames,
            spec_frames_obj.sr,
            spec_frames_obj.window,
            spec_frames_obj.frame_len,
            spec_frames_obj.hop_len,
            spec_frames_obj.center,
            spec_frames_obj.pad_mode,
            lower,
            upper,
        )
