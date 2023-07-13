import librosa
import numpy as np
import pytest

from emvoice.spectral import (
    AlphaRatioFrames,
    HammarIndexFrames,
    MelSpecFrames,
    MfccFrames,
    SpecFrames,
    SpectralFluxFrames,
    SpectralSlopeFrames,
)

from .test_frames import TestBaseFrames


class TestSpecFrames(TestBaseFrames):
    @pytest.fixture
    def spec_frames_obj(self, sig_obj):
        return SpecFrames.from_signal(
            sig_obj, frame_len=self.frame_len, hop_len=self.hop_len
        )

    @pytest.fixture
    def frames_scope(self, spec_frames_obj):
        return spec_frames_obj

    def test_spec(self, spec_frames_obj):
        spec = spec_frames_obj.frames
        assert (
            spec.shape[:1]
            == spec_frames_obj.ts.shape
            == spec_frames_obj.idx.shape
        )
        assert np.all(
            np.iscomplex(spec[:, 1:-1])
        )  # First and last columns are not complex


class TestAlphaRatioFrames(TestSpecFrames):
    @pytest.fixture
    def alpha_ratio_frames_obj(self, spec_frames_obj):
        return AlphaRatioFrames.from_spec_frames(spec_frames_obj)

    @pytest.fixture
    def frames_scope(self, alpha_ratio_frames_obj):
        return alpha_ratio_frames_obj

    def test_alpha_ratio(self, alpha_ratio_frames_obj):
        alpha_ratio = alpha_ratio_frames_obj.frames
        assert (
            alpha_ratio.shape
            == alpha_ratio_frames_obj.ts.shape
            == alpha_ratio_frames_obj.idx.shape
        )
        assert np.all(
            np.logical_or(10 ** (alpha_ratio / 10) > 0, np.isnan(alpha_ratio))
        )


class TestHammarIndexFrames(TestSpecFrames):
    @pytest.fixture
    def hammar_index_frames_obj(self, spec_frames_obj):
        return HammarIndexFrames.from_spec_frames(spec_frames_obj)

    @pytest.fixture
    def frames_scope(self, hammar_index_frames_obj):
        return hammar_index_frames_obj

    def test_hammar_index(self, hammar_index_frames_obj):
        hammar_index = hammar_index_frames_obj.frames
        assert (
            hammar_index.shape
            == hammar_index_frames_obj.ts.shape
            == hammar_index_frames_obj.idx.shape
        )
        assert np.all(
            np.logical_or(10 ** (hammar_index / 10) > 0, np.isnan(hammar_index))
        )


class TestSpectralSlopeFrames(TestSpecFrames):
    @pytest.fixture
    def spectral_slope_frames_obj(self, spec_frames_obj):
        return SpectralSlopeFrames.from_spec_frames(spec_frames_obj)

    @pytest.fixture
    def frames_scope(self, spectral_slope_frames_obj):
        return spectral_slope_frames_obj

    def test_spectral_slope(self, spectral_slope_frames_obj):
        spectral_slope = spectral_slope_frames_obj.frames
        assert (
            spectral_slope.shape[:1]
            == spectral_slope_frames_obj.ts.shape
            == spectral_slope_frames_obj.idx.shape
        )

    def test_calc_spectral_slope(self, spectral_slope_frames_obj):
        n = 512
        band_power = np.random.uniform(0, 1, n)
        band_freqs = np.random.uniform(0, 8000, n)
        coefs = spectral_slope_frames_obj._calc_spectral_slope(
            band_power, band_freqs
        )
        assert coefs.shape == (1,)

    def test_calc_spectral_slope_nan(self, spectral_slope_frames_obj):
        n = 512
        band_power = np.random.uniform(0, 1, n)
        band_freqs = np.random.uniform(0, 8000, n)
        band_power[1:3] = np.nan
        coefs = spectral_slope_frames_obj._calc_spectral_slope(
            band_power, band_freqs
        )
        assert coefs.shape == (1,)


class TestMelSpecFrames(TestSpecFrames):
    @pytest.fixture
    def mel_spec_frames_obj(self, spec_frames_obj):
        return MelSpecFrames.from_spec_frames(spec_frames_obj)

    @pytest.fixture
    def frames_scope(self, mel_spec_frames_obj):
        return mel_spec_frames_obj

    def test_spectral_slope(self, mel_spec_frames_obj):
        mel_spec = mel_spec_frames_obj.frames
        assert (
            mel_spec.shape[:1]
            == mel_spec_frames_obj.ts.shape
            == mel_spec_frames_obj.idx.shape
        )
        assert mel_spec.shape[1] == mel_spec_frames_obj.n_mels


class TestMfccFrames(TestMelSpecFrames):
    @pytest.fixture
    def mfcc_frames_obj(self, mel_spec_frames_obj):
        return MfccFrames.from_mel_spec_frames(mel_spec_frames_obj)

    @pytest.fixture
    def frames_scope(self, mfcc_frames_obj):
        return mfcc_frames_obj

    def test_mfcc(self, mfcc_frames_obj):
        mfcc = mfcc_frames_obj.frames
        assert (
            mfcc.shape[:1]
            == mfcc_frames_obj.ts.shape
            == mfcc_frames_obj.idx.shape
        )
        assert mfcc.shape[1] == mfcc_frames_obj.n_mfcc


class TestSpectralFluxFrames(TestSpecFrames):
    @pytest.fixture
    def spectral_flux_frames_obj(self, spec_frames_obj):
        return SpectralFluxFrames.from_spec_frames(spec_frames_obj)

    @pytest.fixture
    def frames_scope(self, spectral_flux_frames_obj):
        return spectral_flux_frames_obj

    def test_idx(self, sig_obj, frames_scope):
        idx = frames_scope.idx
        assert np.all(
            idx == np.arange((sig_obj.sig.shape[0] + 1) // self.hop_len)
        )

    def test_spectral_flux(self, spectral_flux_frames_obj):
        spectral_flux = spectral_flux_frames_obj.frames

        assert (
            spectral_flux.shape
            == spectral_flux_frames_obj.ts.shape
            == spectral_flux_frames_obj.idx.shape
        )
        assert np.all(spectral_flux > 0)
