import matplotlib.pyplot as plt
import numpy as np

from scipy.fftpack import fft, ifft
from scipy.io import wavfile
from scipy.signal import wiener


def main():
    wav_file = 'LL15'

    rate, data = wavfile.read('{}.wav'.format(wav_file))
    dtype_pcm = data.dtype
    data = pcm2fp(data)
    data_len = round(data.shape[0] / rate)
    data_fft = fft(data, n=int(data_len * rate))

    data_filtered = signal_filtered(rate, data_fft)

    plt.plot(np.abs(data_fft), label='Noisy signal')
    plt.plot(np.abs(data_filtered), label='Filtered signal')
    plt.legend(loc='upper center')

    wavfile.write(
        '{}.wav'.format(wav_file + '_filtered'),
        rate, fp2pcm(data_filtered, dtype_pcm))
    plt.show()


def signal_filtered(rate, data_fft):
    frame_len = int(2 * round(rate / 200))
    return wiener_filter(data_fft, frame_len)


def norm4fp(dtype):
    if dtype == 'int8':
        return 2**8 - 1
    elif dtype == 'int16':
        return 2**15
    elif dtype == 'int32':
        return 2**31
    else:
        raise TypeError(
            'data_pcm.dtype {} != {} or != {} or != {}'.format(
                data_pcm.dtype, 'int8', 'int16', 'int32'))


def pcm2fp(data_pcm):
    return data_pcm.astype(np.float32, order='C') / norm4fp(data_pcm.dtype)


def fp2pcm(data_fp, dtype_pcm):
    data_pcm = np.around(np.real(ifft(data_fp)) * norm4fp(dtype_pcm))
    return data_pcm.astype(dtype_pcm, order='C')


def wiener_filter(data, window_len, noise=0.5):
    return wiener(data, window_len, noise=noise)


if __name__ == '__main__':
    main()
