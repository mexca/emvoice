Examples
========

This section gives a few examples of how emvoice can be used to compute voice features.


Loading and framing a speech signal
-----------------------------------

In the first example, the goal is to load a speech signal and split it into frames,
which are short overlapping slices of the signal.

>>> import librosa
>>> from emvoice.frames import BaseFrames
>>> from emvoice.signal import AudioSignal
>>>
>>> # Get path to example speech file
>>> path_to_example = librosa.example("libri1")
>>>
>>> # Load speech signal from file
>>> audio_signal = AudioSignal.from_file(filename=path_to_example)
>>>
>>> # Divide signal into frames with 1024 samples each
>>> audio_frames = BaseFrames.from_signal(sig_obj=audio_signal, frame_len=1024)

The framed signal can be accessed in the ``BaseFrames.frames`` attribute.

>>> print(audio_frames.frames.shape)
(1279, 1024)


Estimating voice pitch
----------------------

This example demonstrates how to estimate the voice pitch as the fundamental frequency F0
for a framed speech signal.

>>> import librosa
>>> from emvoice.pitch import PitchFrames
>>> from emvoice.signal import AudioSignal
>>>
>>> # Get path to example speech file
>>> path_to_example = librosa.example("libri1")
>>>
>>> # Load speech signal from file
>>> audio_signal = AudioSignal.from_file(filename=path_to_example)
>>>
>>> # Estimate fundamental frequency from signal
>>> pitch_frames = PitchFrames.from_signal(sig_obj=audio_signal, frame_len=1024)
>>>
>>> print(pitch_frames.frames.shape)
(1279,)

The fundamental frequency is stored as a single value for each frame.
The ``PitchFrames.from_signal()`` method automatically frames the audio signal.
The ``PitchFrames.prob`` attribute contains probabilities for each frame that it is voiced (F0 > 0).

>>> print(pitch_frames.prob.shape)
(1279,)


Estimating jitter and shimmer
-----------------------------

Jitter and shimmer can be estimated by first estimating the glottal pulses in a signal.

>>> import librosa
>>> from emvoice.pitch import JitterFrames, PitchFrames, PitchPulseFrames, ShimmerFrames
>>> from emvoice.signal import AudioSignal
>>>
>>> # Get path to example speech file
>>> path_to_example = librosa.example("libri1")
>>>
>>> # Load speech signal from file
>>> audio_signal = AudioSignal.from_file(filename=path_to_example)
>>>
>>> # Estimate fundamental frequency from signal
>>> pitch_frames = PitchFrames.from_signal(sig_obj=audio_signal, frame_len=1024)
>>>
>>> # Estimate pulse frames from signal and pitch
>>> pulse_frames = PitchPulseFrames.from_signal_and_pitch_frames(sig_obj=audio_signal, pitch_frames_obj=pitch_frames)
>>>
>>> print(pulse_frames.frames[250][:2])
[(2.9026303854875284, 0.013333333333333334, 0.011052566), (2.913424036281179, 0.013333333333333334, 0.011512194)]

Each pulse frame contains a list of pulses which are (timestamp, 1/F0, amplitude) tuples.

The ``PitchPulseFrames`` object can be converted into ``JitterFrames`` and ``ShimmerFrames``.

>>> jitter_frames = JitterFrames.from_pitch_pulse_frames(pulse_frames)
>>>
>>> shimmer_frames = ShimmerFrames.from_pitch_pulse_frames(pulse_frames)


Estimating formants
-------------------

Formant frequencies and bandwidths can be estimated from a
special ``FormantSignal`` (also from ``AudioSignal``) to imitate the implementation in Praat.

>>> import librosa
>>> from emvoice.frames import BaseFrames
>>> from emvoice.formants import FormantFrames
>>> from emvoice.signal import AudioSignal, FormantAudioSignal
>>>
>>> # Get path to example speech file
>>> path_to_example = librosa.example("libri1")
>>>
>>> # Load speech signal from file
>>> audio_signal = AudioSignal.from_file(filename=path_to_example)
>>>
>>> # Compute formant signal from audio signal
>>> formant_audio_signal = FormantAudioSignal.from_audio_signal(audio_signal)
>>>
>>> # Frame formant signal
>>> formant_audio_frames = BaseFrames.from_signal(sig_obj=formant_audio_signal, frame_len=1024)
>>>
>>> # Estimate formants from framed formant signal
>>> formant_frames = FormantFrames.from_frames(formant_audio_frames)
>>>
>>> print(formant_frames.frames[250])
[(2158.808975628817, 344.96113877579666), (3879.663445820748, 250.7392916813846)]

Each formant frame contains a list of the formants (here 2) which are (frequency, bandwidth) tuples.
