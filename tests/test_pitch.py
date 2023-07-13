import numpy as np
import pytest

from emvoice.pitch import (
    JitterFrames,
    PitchFrames,
    PitchHarmonicsFrames,
    PitchPeriodFrames,
    PitchPulseFrames,
    ShimmerFrames,
)
from emvoice.signal import AudioSignal, BaseSignal

from .test_frames import TestBaseFrames
from .test_spectral import TestSpecFrames


class TestPitchFrames(TestBaseFrames):
    @pytest.fixture
    def pitch_frames_obj(self, sig_obj):
        return PitchFrames.from_signal(
            sig_obj, frame_len=self.frame_len, hop_len=self.hop_len
        )

    @pytest.fixture
    def frames_scope(self, pitch_frames_obj):
        return pitch_frames_obj

    def test_pitch_pyin(self, pitch_frames_obj):
        pitch_f0 = pitch_frames_obj.frames
        assert (
            pitch_f0.shape
            == pitch_frames_obj.ts.shape
            == pitch_frames_obj.idx.shape
        )
        assert np.all(np.logical_or(pitch_f0 > 0, np.isnan(pitch_f0)))


class TestPitchPulseFrames(TestPitchFrames):
    @pytest.fixture
    def pulses_frames_obj(self, sig_obj, pitch_frames_obj):
        return PitchPulseFrames.from_signal_and_pitch_frames(
            sig_obj, pitch_frames_obj
        )

    @pytest.fixture
    def frames_scope(self, pulses_frames_obj):
        return pulses_frames_obj

    def test_pulses(self, pulses_frames_obj):
        pulses = pulses_frames_obj.frames
        assert isinstance(pulses, list)
        assert (
            len(pulses)
            == pulses_frames_obj.ts.shape[0]
            == pulses_frames_obj.idx.shape[0]
        )
        assert np.all(
            np.array(
                [
                    isinstance(puls, tuple) and np.all(np.array(puls[:2]) >= 0)
                    for frame in pulses
                    for puls in frame
                    if len(frame) > 0
                ]
            )
        )

    def test_get_next_pulse(self, pulses_frames_obj):
        frame = np.random.uniform(-1, 1, self.frame_len)
        ts = np.linspace(0, 1, num=self.frame_len)
        t0 = 0.005 + np.random.uniform(0, 0.0005, self.frame_len)
        start = 0.5 - 0.0025
        stop = 0.5 + 0.0025
        pulses = []
        pulses_frames_obj._get_next_pulse(
            frame, ts, t0, start, stop, True, pulses
        )
        pulses_frames_obj._get_next_pulse(
            frame, ts, t0, start, stop, False, pulses
        )
        assert len(pulses) > 0
        assert np.all(
            np.array(
                [
                    isinstance(puls, tuple) and puls[0] >= 0 and puls[1] >= 0
                    for puls in pulses
                ]
            )
        )


class TestPitchHarmonicsFrames(TestPitchFrames, TestSpecFrames):
    n_harmonics = 100

    @pytest.fixture
    def harmonics_frames_obj(self, spec_frames_obj, pitch_frames_obj):
        return PitchHarmonicsFrames.from_spec_and_pitch_frames(
            spec_frames_obj, pitch_frames_obj, self.n_harmonics
        )

    @pytest.fixture
    def frames_scope(self, harmonics_frames_obj):
        return harmonics_frames_obj

    def test_harmonics(self, harmonics_frames_obj):
        harmonics = harmonics_frames_obj.frames
        assert (
            harmonics.shape[0]
            == harmonics_frames_obj.ts.shape[0]
            == harmonics_frames_obj.idx.shape[0]
        )
        assert np.all(np.logical_or(harmonics >= 0, np.isnan(harmonics)))


class TestJitterFrames(TestPitchPulseFrames):
    @pytest.fixture
    def jitter_frames_obj(self, pulses_frames_obj):
        return JitterFrames.from_pitch_pulse_frames(pulses_frames_obj)

    @pytest.fixture
    def frames_scope(self, jitter_frames_obj):
        return jitter_frames_obj

    def test_jitter(self, jitter_frames_obj):
        jitter = jitter_frames_obj.frames
        assert (
            jitter.shape
            == jitter_frames_obj.ts.shape
            == jitter_frames_obj.idx.shape
        )
        assert np.all(np.logical_or(jitter > 0, np.isnan(jitter)))

    def test_calc_jitter_frame(self, pulses_frames_obj, jitter_frames_obj):
        jitter_frame = jitter_frames_obj._calc_jitter_frame(
            pulses_frames_obj.frames[1],
            rel=True,
            lower=0.0001,
            upper=0.02,
            max_period_ratio=1.3,
        )
        assert isinstance(jitter_frame, float)
        assert jitter_frame > 0 or np.isnan(jitter_frame)


class TestShimmerFrames(TestPitchPulseFrames):
    @pytest.fixture
    def shimmer_frames_obj(self, pulses_frames_obj):
        return ShimmerFrames.from_pitch_pulse_frames(pulses_frames_obj)

    @pytest.fixture
    def frames_scope(self, shimmer_frames_obj):
        return shimmer_frames_obj

    def test_shimmer(self, shimmer_frames_obj):
        shimmer = shimmer_frames_obj.frames
        assert (
            shimmer.shape
            == shimmer_frames_obj.ts.shape
            == shimmer_frames_obj.idx.shape
        )
        assert np.all(np.logical_or(shimmer > 0, np.isnan(shimmer)))

    def test_calc_shimmer_frame(self, pulses_frames_obj, shimmer_frames_obj):
        shimmer_frame = shimmer_frames_obj._calc_shimmer_frame(
            pulses_frames_obj.frames[1],
            rel=True,
            lower=0.0001,
            upper=0.02,
            max_period_ratio=1.3,
            max_amp_factor=1.6,
        )
        assert isinstance(shimmer_frame, float)
        assert shimmer_frame > 0 or np.isnan(shimmer_frame)

    def test_get_amplitude(self, pulses_frames_obj, shimmer_frames_obj):
        _, mask = shimmer_frames_obj._calc_period_length(
            pulses_frames_obj.frames[1], 0.0001, 0.02
        )
        amps = shimmer_frames_obj._get_amplitude(
            pulses_frames_obj.frames[1], mask
        )
        assert isinstance(amps, list)
