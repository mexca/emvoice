import librosa
import numpy as np
import pytest

from emvoice.formants import FormantAmplitudeFrames, FormantFrames

from .test_frames import TestBaseFrames
from .test_pitch import TestPitchHarmonicsFrames


class TestFormantFrames(TestBaseFrames):
    @pytest.fixture
    def formant_frames_obj(self, sig_frames_obj):
        return FormantFrames.from_frames(sig_frames_obj)

    @pytest.fixture
    def frames_scope(self, formant_frames_obj):
        return formant_frames_obj

    def test_formants(self, formant_frames_obj):
        formants = formant_frames_obj.frames
        assert isinstance(formants, list)
        assert (
            len(formants)
            == formant_frames_obj.ts.shape[0]
            == formant_frames_obj.idx.shape[0]
        )
        assert np.all(np.array([np.all(np.array(f) > 0) for f in formants]))

    def test_calc_formants(self, sig_frames_obj, formant_frames_obj):
        coefs = librosa.lpc(sig_frames_obj.frames, order=10)
        formants = formant_frames_obj._calc_formants(
            coefs[0, :],
            sig_frames_obj.sr,
            formant_frames_obj.lower,
            formant_frames_obj.upper,
        )
        assert isinstance(formants, list)
        assert np.all(np.array(formants))


class TestFormantAmplitudeFrames(TestFormantFrames, TestPitchHarmonicsFrames):
    @pytest.fixture
    def formant_amp_frames_obj(
        self, formant_frames_obj, harmonics_frames_obj, pitch_frames_obj
    ):
        return FormantAmplitudeFrames.from_formant_harmonics_and_pitch_frames(
            formant_frames_obj, harmonics_frames_obj, pitch_frames_obj
        )

    @pytest.fixture
    def frames_scope(self, formant_amp_frames_obj):
        return formant_amp_frames_obj

    def test_formant_amplitude(
        self, formant_amp_frames_obj, formant_frames_obj
    ):
        amp = formant_amp_frames_obj.frames
        assert isinstance(amp, np.ndarray)
        assert (
            amp.shape[0]
            == formant_amp_frames_obj.ts.shape[0]
            == formant_amp_frames_obj.idx.shape[0]
        )
        assert amp.shape[1] == formant_frames_obj.max_formants
