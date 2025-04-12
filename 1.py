import numpy as np
import matplotlib.pyplot as plt

T = 1.0
fo = 1
fn = 100
t = np.linspace(-T, T, 128)

video_impulse = np.where((t >= -T/2) & (t <= T/2), 1, 0.0)
low_amplitude_video_impulse = np.where((t >= -T/2) & (t <= T/2), 0.5, 0.0)
low_range_video_impulse = np.where((t >= -T/4) & (t <= T/4), 1, 0.0)
modulated = (1 + 0.5 * np.sin(2 * np.pi * fo * t)) * np.sin(2 * np.pi * fn * t)
radio_impulse = video_impulse * np.sin(2 * np.pi * fn * t)

spectrum = np.fft.fftshift(np.fft.fft(video_impulse))
spectrum_low_amplitude = np.fft.fftshift(np.fft.fft(low_amplitude_video_impulse))
spectrum_low_range = np.fft.fftshift(np.fft.fft(low_range_video_impulse))
spectrum_modulated = np.fft.fftshift(np.fft.fft(modulated))
spectrum_radio = np.fft.fftshift(np.fft.fft(radio_impulse))
freq = np.fft.fftshift(np.fft.fftfreq(128))
plt.figure(figsize=(12, 6))

plt.subplot(3, 4, 1)
plt.plot(t, video_impulse)
plt.xlabel("Амплитуда=1, длительность=T/2")
plt.title("Видеоимпульс")

plt.subplot(3, 4, 2)
plt.plot(freq, np.real(spectrum))
plt.xlabel("Амплитуда=1, длительность=T/2")
plt.title("Спектр")

plt.subplot(3, 4, 3)
plt.plot(t, low_amplitude_video_impulse)
plt.xlabel("Амплитуда=0.5, длительность=T/2")
plt.title("Видеоимпульс")

plt.subplot(3, 4, 4)
plt.plot(freq, np.real(spectrum_low_amplitude))
plt.xlabel("Амплитуда=0.5, длительность=T/2")
plt.title("Спектр")

plt.subplot(3, 4, 5)
plt.plot(t, low_range_video_impulse)
plt.xlabel("Амплитуда=1, длительность=T/4")
plt.title("Видеоимпульс")

plt.subplot(3, 4, 6)
plt.plot(freq, np.real(spectrum_low_range))
plt.xlabel("Амплитуда=1, длительность=T/4")
plt.title("Спектр")

plt.subplot(3, 4, 9)
plt.plot(t, modulated)
plt.title("АМ-сигнал")

plt.subplot(3, 4, 10)
plt.plot(freq, np.abs(spectrum_modulated))
plt.title("Спектр АМ-сигнала")

# Третья строка: радиоимпульс
plt.subplot(3, 4, 11)
plt.plot(t, radio_impulse)
plt.title("Радиоимпульс")

plt.subplot(3, 4, 12)
plt.plot(freq, np.real(spectrum_radio))
plt.title("Спектр радиоимпульса")

plt.tight_layout()
plt.show()