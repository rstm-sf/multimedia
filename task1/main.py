import matplotlib.pyplot as plt
import numpy as np

from scipy.fftpack import fft
from scipy.io import wavfile


def main():
    wav_file = 'L15No.wav'
    rate, data = wavfile.read(wav_file)
    data = pcm2fp(data)
    data_len = round(data.shape[0] / rate)

    data_fft = fft(data, n=int(data_len * rate))

    plt.plot(np.abs(data_fft))
    plt.show()


def pcm2fp(data_pcm):
    if data_pcm.dtype == 'int8':
        _max = 2**8 - 1
    elif data_pcm.dtype == 'int16':
        _max = 2**15
    elif data_pcm.dtype == 'int32':
        _max = 2**31
    else:
        raise TypeError(
            'data_pcm.dtype {} != {} or != {} or != {}'.format(
                data_pcm.dtype, 'int8', 'int16', 'int32'))
    return data_pcm.astype(np.float32, order='C') / _max


if __name__ == '__main__':
    main()
