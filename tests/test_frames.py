import librosa
import numpy as np
import pytest

from emvoice.frames import BaseFrames
from emvoice.signal import AudioSignal, BaseSignal


class TestBaseFrames:
    frame_len = 1024
    hop_len = frame_len // 4

    @pytest.fixture
    def audio_file(self):
        return librosa.ex("trumpet")

    @pytest.fixture
    def sig_obj(self, audio_file):
        return AudioSignal.from_file(audio_file)

    @pytest.fixture
    def sig_frames_obj(self, sig_obj):
        return BaseFrames.from_signal(
            sig_obj, frame_len=self.frame_len, hop_len=self.hop_len
        )

    @pytest.fixture
    def frames_scope(self, sig_frames_obj):
        return sig_frames_obj

    def test_idx(self, sig_obj, frames_scope):
        idx = frames_scope.idx
        assert np.all(
            idx == np.arange((sig_obj.sig.shape[0] + 1) // self.hop_len + 1)
        )

    def test_ts(self, frames_scope):
        ts = frames_scope.ts
        assert np.all(
            ts
            == librosa.frames_to_time(
                frames_scope.idx,
                sr=frames_scope.sr,
                hop_length=frames_scope.hop_len,
            )
        )
