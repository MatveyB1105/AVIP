from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np


def spectrogram_plot(samples, sample_rate, t=10000):
    frequencies, times, my_spectrogram = signal.spectrogram(samples, sample_rate, scaling='spectrum', window='hann')
    spec = np.log10(my_spectrogram + 1e-10)
    plt.pcolormesh(times, frequencies, spec, shading='gouraud', vmin=spec.min(), vmax=spec.max(),cmap='inferno')

    plt.ylim(top = t)
    plt.ylabel('Частота [Гц]')
    plt.xlabel('Время [с]')


def denoise(samples, sample_rate, cutoff_freuency, passes=1):
    z = signal.savgol_filter(samples, 100, 3)
    b, a = signal.butter(3, cutoff_freuency / sample_rate)
    zi = signal.lfilter_zi(b, a)
    for _ in range(passes):
        z, _ = signal.lfilter(b, a, z, zi=zi * z[0])
    return z


def to_pcm(y):
    return np.int16(y / np.max(np.abs(y)) * 32000)


def find_high_energy_windows(samples, sample_rate, window_duration=0.1):
    nperseg = int(sample_rate * window_duration)

    frequencies, times, Sxx = signal.spectrogram(
        samples,
        sample_rate,
        scaling='spectrum',
        window='hann',
        nperseg=nperseg,
        noverlap=0
    )

    # Энергия = сумма по всем частотам
    energy_total = np.sum(Sxx, axis=0)

    # Находим окно с максимальной энергией
    max_energy_index = np.argmax(energy_total)
    max_time = times[max_energy_index]

    print(f"[INFO] Самая высокая суммарная энергия наблюдается на моменте {max_time:.2f} секунд.")
    return max_time, times, energy_total



if __name__ == '__main__':
    dpi = 500

    sample_rate, samples = wavfile.read('src/gitara-grust-besplatno-film-fonovyiy-zvuk-40707.wav')
    if samples.ndim > 1:
        samples = samples[:, 0]  # если стерео, берём только один канал

    plt.figure(dpi=dpi)
    spectrogram_plot(samples, sample_rate, 20000)
    plt.savefig('results/spectrogram.png', dpi=dpi)
    plt.clf()

    denoised_0 = denoise(samples, sample_rate, cutoff_freuency=1000, passes=0)
    spectrogram_plot(denoised_0, sample_rate, 20000)
    plt.savefig('results/denoised_spectrogram_savgol.png', dpi=dpi)
    plt.clf()

    denoised = denoise(samples, sample_rate, cutoff_freuency=1000)
    spectrogram_plot(denoised, sample_rate)
    plt.savefig('results/denoised_spectrogram_once.png', dpi=dpi)
    plt.clf()

    wavfile.write('results/denoised_once.wav', sample_rate, to_pcm(denoised))

    denoised_2 = denoise(samples, sample_rate, cutoff_freuency=1000, passes=2)
    spectrogram_plot(denoised_2, sample_rate)
    plt.savefig('results/denoised_spectrogram_twice.png', dpi=dpi)
    plt.clf()

    wavfile.write('results/denoised_twice.wav', sample_rate, to_pcm(denoised_2))

    # Анализ энергии
    max_time, times, energy_total = find_high_energy_windows(samples, sample_rate)
    plt.plot(times, energy_total)
    plt.title('Общая энергия сигнала по времени')
    plt.xlabel('Время [с]')
    plt.ylabel('Энергия')
    plt.grid(True)
    plt.savefig('results/energy_peaks.png', dpi=500)
    plt.clf()

