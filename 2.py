import numpy as np
import matplotlib.pyplot as plt

mod_schemes = ['BPSK','QPSK','16QAM']
Nsc = 2048
SNR = 1

def modulate(scheme):
    mod_map_coordinates = []
    if scheme == 'BPSK':
        bits = np.random.randint(0, 2, Nsc * 1)
        for i in range(0,len(bits)):
            if bits[i] == 0:
                mod_map_coordinates.append(1)
            else:
                mod_map_coordinates.append(-1)
        return np.array(mod_map_coordinates)
    elif scheme == 'QPSK':
        bits = np.random.randint(0, 2, Nsc * 2)
        mod_alg = {(0,0): (1 + 1j), (0,1): (-1 + 1j), (1,1): (-1 - 1j), (1,0): (1 - 1j)}
        for i in range(0, len(bits), 2):
            bit_pair = (bits[i],bits[i+1])
            coordinate = mod_alg[bit_pair]
            mod_map_coordinates.append(coordinate)
        return np.array(mod_map_coordinates)
    elif scheme == '16QAM':
        bits = np.random.randint(0, 2, Nsc * 4)
        mod_alg = {(0,0): -3, (0,1): -1, (1,1): 1, (1,0): 3}
        for i in range(0, len(bits), 4):
            bit_pair_i = (bits[i],bits[i+1])
            bit_pair_q = (bits[i+2],bits[i+3])
            i_val = mod_alg[bit_pair_i]
            q_val = mod_alg[bit_pair_q]
            mod_map_coordinates.append((i_val + 1j*q_val))
        return np.array(mod_map_coordinates)

def FFT(signal):
    return np.fft.fftshift(np.fft.fft(signal))

def IFFT(spectrum):
    return np.fft.ifft(np.fft.ifftshift(spectrum))

def add_noise(signal, SNR_dB):
    SNR_linear = 10 ** (SNR_dB / 10)
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / SNR_linear
    noise = (np.random.randn(len(signal)) * np.sqrt(noise_power / 2)) + 1j * (np.random.randn(len(signal)) * np.sqrt(noise_power / 2))
    return np.array(signal) + np.array(noise)

for mod_scheme in mod_schemes:
    mod_symbols = modulate(mod_scheme)
    OFDM_signal = IFFT(mod_symbols)
    signal_with_noise = add_noise(OFDM_signal, SNR)
    demod_OFDM_signal = FFT(signal_with_noise)

    plt.figure(figsize=(7, 7))
    plt.suptitle(f'Вид модуляции - {mod_scheme}, SNR = {SNR} дБ')

    plt.scatter(demod_OFDM_signal.real, demod_OFDM_signal.imag, color='green', s=10, alpha=0.6)
    plt.scatter(mod_symbols.real, mod_symbols.imag, color='red', s=10)
    plt.title('Принятый сигнал - зеленый, отправленный - красный')
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.grid(True)
    plt.axhline(0, color='gray', lw=1)
    plt.axvline(0, color='gray', lw=1)
    plt.axis('equal')

    plt.tight_layout()
    plt.show()
