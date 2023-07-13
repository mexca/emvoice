import librosa
import numpy as np
import pytest

from emvoice.signal import AudioSignal, BaseSignal


@pytest.fixture
def audio_file():
    return librosa.ex("trumpet")


class TestBaseSignal:
    n_samples = 10

    @pytest.fixture
    def sig_obj(self):
        return BaseSignal(
            np.random.uniform(-1, 1, self.n_samples), self.n_samples
        )

    def test_idx(self, sig_obj):
        idx = sig_obj.idx
        assert np.all(idx == np.arange(self.n_samples))

    def test_ts(self, sig_obj):
        ts = sig_obj.ts
        assert np.all(ts == sig_obj.idx / sig_obj.sr)


class TestAudioSignal(TestBaseSignal):
    def test_from_file(self, audio_file):
        sig_obj = AudioSignal.from_file(audio_file)
        assert isinstance(sig_obj, (AudioSignal, BaseSignal))
