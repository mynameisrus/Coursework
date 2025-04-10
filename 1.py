import numpy as np
import matplotlib.pyplot as plt

T = 1.0
fo = 1
fn = 100
t = np.linspace(-T, T, 128)

video_impulse = np.where((t >= -T/2) & (t <= T/2), 1, 0.0)
modulated = (1 + 0.5 * np.sin(2 * np.pi * fo * t)) * np.sin(2 * np.pi * fn * t)

spectrum = np.fft.fftshift(np.fft.fft(video_impulse))
freq = np.fft.fftshift(np.fft.fftfreq(128))

spectrum_modulated = np.fft.fftshift(np.fft.fft(modulated))
freq_modulated = np.fft.fftshift(np.fft.fftfreq(128))

plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1)
plt.plot(t, video_impulse)
plt.title("Прямоугольный видеоимпульс")

plt.subplot(1, 4, 2)
plt.plot(freq, np.real(spectrum))
plt.title("Спектр прямоугольного видеоимпульса")

plt.subplot(1, 4, 3)
plt.plot(t, modulated)
plt.title("АМ-сигнал")

plt.subplot(1, 4, 4)
plt.plot(freq_modulated, np.abs(spectrum_modulated))
plt.title("Спектр АМ-сигнала")

plt.tight_layout()
plt.show()