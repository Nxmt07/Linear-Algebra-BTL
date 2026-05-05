import numpy as np
import librosa
import soundfile as sf

def manual_fft(x):
    N = len(x)
    if N <= 1:
        return x
    even = manual_fft(x[0::2])
    odd = manual_fft(x[1::2])
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + \
           [even[k] - T[k] for k in range(N // 2)]

def manual_ifft(X):
    N = len(X)
    X_conj = np.conjugate(X)
    y = manual_fft(X_conj)
    return np.conjugate(y) / N

def pad_to_power_of_2(x):
    n = len(x)
    next_pow2 = 1 << (n - 1).bit_length()
    return np.pad(x, (0, next_pow2 - n), mode='constant')

def process_audio(input_path, output_path, threshold_ratio=0.02):
    y, sr = librosa.load(input_path, sr=None, mono=True)
    original_len = len(y)
    y_padded = pad_to_power_of_2(y)
    
    freq_domain = np.array(manual_fft(y_padded))
    
    magnitudes = np.abs(freq_domain)
    limit = np.max(magnitudes) * threshold_ratio
    freq_domain[magnitudes < limit] = 0
    
    y_denoised = np.real(manual_ifft(freq_domain))
    y_final = y_denoised[:original_len]
    
    sf.write(output_path, y_final, sr)

if __name__ == "__main__":
    process_audio('input.mp3', 'output.mp3')