Algorithms
==========

This section describes the algorithms that are used in emvoice.
Many algorithms used are implemented in other packages, to which we will refer in this case.


Loading speech signals
----------------------

emvoice uses :func:`librosa.load` for loading speech signals.
It supports all codes that are supported by ``soundfile`` or ``audioread``.
For resampling, it uses the high-quality mode in ``soxr``.


Voice pitch
----------------------

emvoice uses :func:`librosa.pyin` for estimating voice pitch as the fundamental frequency.
The details of this algorithm are described in [Mauch2014]_.


Pitch harmonics
--------------------------

The harmonics of the fundamental frequency are estimated by linear interpolation as in :func:`librosa.f0_harmonics`.


Glottal pulses
-------------------------

Glottal pulses to estimate jitter and shimmer are computed using the point process algorithm
implemented in `Praat <https://github.com/praat/praat/blob/master/fon/Sound_to_PointProcess.cpp>`_:

1. Interpolate the fundamental frequency at the timestamps of the framed (padded) signal.
2. Start at the mid point `m` of each frame and create an interval [start, stop], where ``start=m-T0/2`` and ``stop=m+T0/2`` and `T0` is the fundamental period (1/F0).
3. Detect pulses in the interval by:

    a. Find the maximum amplitude in an interval within the frame.
    b. Compute the fundamental period `T0_new` at the timestamp of the maximum `m_new`.

4. Shift the interval recursively to the right or left until the edges of the frame are reached:

    a. When shifting to the left, set ``start_new=m_new-1.25*T0_new`` and ``stop_new=m_new-0.8*T0_new``.
    b. When shifting to the right, set ``start_new=m_new+0.8*T0_new`` and ``stop_new=m_new+1.25*T0_new``.

5. Filter out duplicate pulses.


Jitter and shimmer
-----------------------------

Jitter is computed as the average absolute difference between consecutive fundamental periods.
Fundamental periods are calculated as the first-order temporal difference between consecutive glottal pulses.

Shimmer is calculated as the average absolute difference between consecutive pitch amplitudes.
Pitch amplitudes are signal amplitudes at the glottal pulses.

The algorithms for both features are adapted from Praat.


Formant frequencies and amplitudes
---------------------------------------------

Formant frequencies are computed using an algorithm adapted from Praat via Burg's method:

1. Apply a preemphasis function with the coefficient ``math.exp(-2 * math.pi * preemphasis_from * (1 / sr))`` to the signal.
2. Apply a window function to the signal. By default, the same Gaussian window as in Praat is used: ``(np.exp(-48.0 * (n - ((N + 1)/2)**2 / (N + 1)**2) - np.exp(-12.0)) / (1.0 - np.exp(-12.0))``, where `N` is the length of the window and `n` the index of each sample.
3. Calculate linear predictive coefficients using :func:`librosa.lpc` with order ``2 * max_formants``.
4. Find the roots of the coefficients.
5. Compute the formant central frequencies as ``np.abs(np.arctan2(np.imag(roots), np.real(roots))) * sr / (2 * math.pi)``.
6. Compute the formant bandwidth as ``np.sqrt(np.abs(np.real(roots) ** 2) + np.abs(np.imag(roots) ** 2)) * sr / (2 * math.pi)``.
7. Filter out formants outside the lower and upper limits.

Formant amplitudes are estimated as the maximum amplitude of harmonics of the
fundamental frequency within an interval ``[lower*f, upper*f]`` where `f` is the
central frequency of the formant in each frame.


Harmonics-to-noise ratio (HNR)
------------------------------

The HNR is computed via an algorithm by [Boersma1993]_ using the autocorrelation function.


Root mean squared (RMS) energy
------------------------------

The RMS energy or equivalent sound level [Eyben2016]_ is computed via :func:`ibrosa.feature.rms`.


Spectrogram
-----------

Spectrograms are computed via Short-time Fourier Transform (STFT) in :func:`librosa.stft`.
Uses ``numpy``'s Fast Fourier Transform library by default.

Mel frequency spectrograms are computed via :func:`librosa.feature.melspectrogram`.


Mel frequency cepstral coefficients (MFCC)
------------------------------------------

MFCCs are computed using :func:`librosa.feature.mfcc`.


Alpha ratio
-----------

The definition of the alpha ratio is adopted from [Eyben2016]_ and [Patel2010]_: Dividing the energy (sum of magnitude) in the lower frequency band (50-1000 Hz)
by the energy in the upper frequency band (1000-5000 Hz). The ratio is then converted to dB.


Hammarberg index
----------------

The definition of the Hammarberg index is adopted from [Eyben2016]_ and [Hammarberg1980]_: Dividing the peak magnitude in the spectrogram region below a pivot point (2000 Hz)
by the peak magnitude in the region above up to an upper limit (5000 Hz).


Spectral slopes
---------------

Spectral slopes are esimtated by fitting linear models to frequency bands predicting power in dB from frequency in Hz as in [Tamarit2008]_.
Fits separate models for each frame and band.


Spectral flux
-------------

Spectral flux is computed as in openSMILE:

1. Compute the normalized magnitudes of the frame spectra by dividing the magnitude at each frequency bin by the sum of all frequency bins.
2. Compute the first-order difference of normalized magnitudes for each frequency bin within an upper and lower limit across frames.
3. Sum up the squared differences for each frame.


References
----------

.. [Boersma1993] Boersma, P. (1993). Accurate short-term analysis of the fundamental frequency and the harmonics-to-noise ratio of a sampled sound. Proceedings of the Institute of Phonetic Sciences, 17, 97–110.

.. [Eyben2016] Eyben, F., Scherer, K. R., Schuller, B. W., Sundberg, J., Andre, E., Busso, C., Devillers, L. Y., Epps, J., Laukka, P., Narayanan, S. S., & Truong, K. P. (2016). The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing. IEEE Transactions on Affective Computing, 7(2), 190–202. https://doi.org/10.1109/TAFFC.2015.2457417

.. [Hammarberg1980] Hammarberg, B., Fritzell, B., Gaufin, J., Sundberg, J., & Wedin, L. (1980). Perceptual and Acoustic Correlates of Abnormal Voice Qualities. Acta Oto-Laryngologica, 90(1–6), 441–451. https://doi.org/10.3109/00016488009131746

.. [Mauch2014] Mauch, M., & Dixon, S. (2014). PYIN: A fundamental frequency estimator using probabilistic threshold distributions. 2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 659–663. https://doi.org/10.1109/ICASSP.2014.6853678

.. [Patel2010] Patel, S., Scherer, K. R., Sundberg, J., & Björkner, E. (2010). Acoustic Markers of Emotions Based on Voice Physiology. Speech Prosody.

.. [Tamarit2008] Tamarit, L., Goudbeek, M., & Scherer, K. (2008). Spectral Slope Measurements in Emotionally Expressive Speech.
