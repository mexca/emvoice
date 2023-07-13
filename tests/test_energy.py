import librosa
import numpy as np
import pytest

from emvoice.energy import HnrFrames, RmsEnergyFrames

from .test_frames import TestBaseFrames
from .test_spectral import TestSpecFrames


class TestHnrFrames(TestBaseFrames):
    @pytest.fixture
    def hnr_frames_obj(self, sig_frames_obj):
        return HnrFrames.from_frames(sig_frames_obj)

    @pytest.fixture
    def frames_scope(self, hnr_frames_obj):
        return hnr_frames_obj

    def test_hnr(self, hnr_frames_obj):
        hnr = hnr_frames_obj.frames
        assert hnr.shape == hnr_frames_obj.ts.shape == hnr_frames_obj.idx.shape
        assert np.all(np.logical_or(10 ** (hnr / 10) > 0, np.isnan(hnr)))

    def test_find_max_peak(self, hnr_frames_obj):
        sig = 1 + 0.3 * np.sin(
            2 * np.pi * 140 * np.linspace(0, 1, self.frame_len)
        )
        autocor = librosa.autocorrelate(sig)
        max_peak = hnr_frames_obj._find_max_peak(
            autocor, hnr_frames_obj.sr, hnr_frames_obj.lower
        )
        assert (
            max_peak
            == autocor[
                (1 / np.arange(autocor.shape[0]) * hnr_frames_obj.sr)
                < hnr_frames_obj.sr / 2
            ].max()
        )

    def test_find_max_peak_all_below_threshold(self, hnr_frames_obj):
        sig = 1 + 0.3 * np.sin(
            2 * np.pi * 0 * np.linspace(0, 1, self.frame_len)
        )
        autocor = librosa.autocorrelate(sig)
        max_peak = hnr_frames_obj._find_max_peak(
            autocor, hnr_frames_obj.sr, hnr_frames_obj.lower
        )
        assert np.isnan(max_peak)


class TestRmsEnergyFrames(TestSpecFrames):
    @pytest.fixture
    def rms_energy_frames_obj(self, spec_frames_obj):
        return RmsEnergyFrames.from_spec_frames(spec_frames_obj)

    @pytest.fixture
    def frames_scope(self, rms_energy_frames_obj):
        return rms_energy_frames_obj

    def test_rms_energy(self, rms_energy_frames_obj):
        rms_energy = rms_energy_frames_obj.frames

        assert (
            rms_energy.shape
            == rms_energy_frames_obj.ts.shape
            == rms_energy_frames_obj.idx.shape
        )
        assert np.all(10 ** (rms_energy / 10.0) > 0)
