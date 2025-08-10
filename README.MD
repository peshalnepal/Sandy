Here’s a shorter README updated to include `requirements.txt` and your script name.

# README (Short)

## What this script does

* Records 5 seconds of mono audio at 16 kHz.
* Pads to next power of two.
* Computes FFT and plots real/imag parts of the **unique** spectrum.
* Rebuilds the full spectrum with correct Hermitian mirroring.
* Runs inverse FFT and saves the result.

## Files created

* `main.wav` — raw recording
* `output_new.wav` — reconstructed audio

## Setup

```bash
python -m venv .venv
source .venv/bin/activate 

# install dependencies
pip install -r requirements.txt
```

(On Linux you may need PortAudio: `sudo apt install portaudio19-dev`.)

## How to run

```bash
python fourier_transform.py
```

## How it works (very brief)

* Keep unique bins: DC to Nyquist (if N even), else DC to floor(N/2).
* Rebuild by mirroring positive bins in reverse with complex conjugate.

  * Even N: mirror `1 .. N/2-1`
  * Odd N: mirror `1 .. (N-1)/2`
* Inverse FFT → time domain → write WAV.

## Customize

* Edit `duration`, `samplerate`, and `filename` at the top.

## Tips

* No windowing used; add a Hann window if analyzing spectra.
* Convert to `int16` only at the end to avoid clipping.
