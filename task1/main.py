import matplotlib.pyplot as plt
import numpy as np

from scipy.fftpack import fft, ifft
from scipy.io import wavfile
from scipy.signal import butter, firwin, lfilter


def main():
    wav_file = 'L15No'

    rate, data = wavfile.read('{}.wav'.format(wav_file))
    dtype_pcm = data.dtype
    data = pcm2fp(data)
    data_len = round(data.shape[0] / rate)
    data_fft = fft(data, n=int(data_len * rate))

    lowcut = 500.0
    highcut = 1500.0
    data_filtered = firwin_hamming_filter(data_fft, lowcut, highcut, rate)

    plt.plot(np.abs(data_fft), label='Noisy signal')
    plt.plot(np.abs(data_filtered), label='Filtered signal')
    plt.legend(loc='upper center')

    wavfile.write(
        '{}.wav'.format(wav_file + '_filtered'),
        rate, fp2pcm(data_filtered, dtype_pcm))
    plt.show()


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


def butter_bandpass(lowcut, highcut, rate, order=5):
    nyq = rate / 2
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='bandpass')


def butter_bandpass_filter(data, lowcut, highcut, rate, order=5):
    b, a = butter_bandpass(lowcut, highcut, rate, order=order)
    return lfilter(b, a, data)


def firwin_hamming(lowcut, highcut, rate, order=3):
    return firwin(order, [lowcut, highcut], pass_zero=False, fs=(rate / 2))


def firwin_hamming_filter(data, lowcut, highcut, rate, order=3):
    taps = firwin_hamming(lowcut, highcut, rate, order=order)
    return lfilter(taps, 1.0, data)


if __name__ == '__main__':
    main()
