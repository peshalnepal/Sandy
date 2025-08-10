import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import matplotlib.pyplot as plt
import math
# Recording settings
samplerate = 16000  # Hz
duration = 5        # seconds
filename = "output_new.wav"
def dft(wav):
    N = len(wav)
    K = np.arange(N).reshape(N, 1)      # N×1
    n = np.arange(N).reshape(1, N)      # 1×N (optional for clarity)
    value = np.exp(-2j * np.pi * K * n / N)
    return np.dot(value, wav)


def fft(wav):
    N = len(wav)
    if N<=2:
        return dft(wav)
    wav_even=fft(wav[::2])
    wav_odd=fft(wav[1::2])
    twiddle_factor=np.exp(-2j*np.pi*np.arange(N//2)/N)
    top_part=wav_even+twiddle_factor*wav_odd
    bottom_part=wav_even-twiddle_factor*wav_odd
    return np.concatenate([top_part,bottom_part])


def inv_dft(freq):
    N = len(freq)
    n = np.arange(N).reshape(N, 1)      # N×1
    k = np.arange(N).reshape(1, N)      # 1×N
    value = np.exp( 2j * np.pi * n * k / N)
    return np.dot(value, freq)

def inv_fft(freq):
    N = len(freq)
    if N<=2:
        return inv_dft(freq)
    wav_even=inv_fft(freq[::2])
    wav_odd=inv_fft(freq[1::2])
    twiddle_factor=np.exp(2j*np.pi*np.arange(N//2)/N)
    top_part=wav_even+twiddle_factor*wav_odd
    bottom_part=wav_even-twiddle_factor*wav_odd
    return np.concatenate([top_part,bottom_part])

print("Recording...")
audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
sd.wait()  

y = audio_data.ravel() 
write("main.wav", samplerate, y)
target_len = 1 << (len(y) - 1).bit_length()
pad_width = target_len - len(y)   # 6
wave = np.pad(y, (0, pad_width), mode='constant')

freq_range=fft(wave)
number_bins=(len(freq_range)+1)/2 if len(freq_range)%2!=0 else (len(freq_range))/2 +1
number_bins=int(number_bins)
x = np.arange(number_bins)
freq_range=freq_range[:number_bins]

target_len = 1 << (len(y) - 1).bit_length()
freq = np.empty(target_len, dtype=np.complex128)
if target_len % 2 == 0:
    freq[:target_len//2 + 1] = freq_range
    freq[target_len//2 + 1:] = np.conj(freq_range[1:-1][::-1])
else:
    freq[:(target_len + 1)//2] = freq_range
    freq[(target_len + 1)//2:] = np.conj(freq_range[1:][::-1])
crt_audio=inv_fft(freq)/(len(freq))
crt_audio=crt_audio[:len(y)].real
crt_audio = np.asarray(crt_audio, dtype=np.int16)
crt_audio=crt_audio.reshape(len(crt_audio),1)
write(filename, samplerate, crt_audio)

if freq_range.shape[0] != x.shape[0]:
    raise ValueError(f"Length mismatch: x={x.shape[0]} vs freq_range={freq_range.shape[0]}")

# Real part
plt.figure()
plt.plot(x, np.real(freq_range), linewidth=1)
plt.xlabel("Frequency bin (k)")
plt.ylabel("Real part Re{X[k]}")
plt.title("DFT Real Part")
plt.grid(True)

# Imaginary part
plt.figure()
plt.plot(x, np.imag(freq_range), linewidth=1)
plt.xlabel("Frequency bin (k)")
plt.ylabel("Imag part Im{X[k]}")
plt.title("DFT Imaginary Part")
plt.grid(True)

plt.show()
