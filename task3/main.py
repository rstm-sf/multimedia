import math
import matplotlib.pyplot as plt
import numpy as np

from scipy.fftpack import fft, ifft
from scipy.io import wavfile
from scipy.signal import firwin, lfilter, medfilt


def main():
    wav_file = 'female_ana'

    rate, data = wavfile.read('{}.wav'.format(wav_file))
    dtype_pcm = data.dtype
    #data = data[:, 0]
    data = pcm2fp(data)

    pitch_contour = pitch_estimation(data, rate / 2, 700)
    return

    #coef = 1.30
    #len_frame = 4000
    #data_filtered = signal_filtered(data, coef, len_frame)

    wavfile.write(
        '{}.wav'.format(wav_file + '_filtered'),
        rate, fp2pcm(data_filtered, dtype_pcm))


def signal_filtered(data, coef, len_frame):
    b = 0
    e = len_frame
    ln = len(data)
    out = np.zeros(ln)
    r = int(len_frame * coef)
    shift = int((ln - r) * len_frame / (ln - len_frame))
    pos = 0
    while e <= ln:
        data_frame = data[b:e]
        res = stretch(data_frame, coef)
        out[pos:pos + r] += 0.5 * res
        pos += shift
        b += len_frame
        e += len_frame
    return out


def stretch(data_frame, coef):
    len_out = int(len(data_frame) * coef)
    out = np.zeros(len_out)
    for i in range(len_out):
        ind = int(i / coef)
        out[i] = data_frame[ind]
    return out


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
    data_pcm = np.around(data_fp * norm4fp(dtype_pcm))
    return data_pcm.astype(dtype_pcm, order='C')


def lowpass_filter(data, fs, f_cut):
    taps = firwin(3, f_cut, fs=fs)
    return lfilter(data, 1.0, data)


def pitch_estimation(data, fs, f_cut):
    lowpass = lowpass_filter(data, fs, f_cut)

    frame_size = round(fs * 0.03)
    frame_rate = round(fs * 0.01)
    len_data = len(lowpass)
    number_of_frames = math.floor((len_data - frame_size) / frame_rate) + 2
    frame_pitch = np.zeros(number_of_frames + 2, dtype=np.float32)

    shift = 1
    for i in range(1, number_of_frames + 1):
        frame_pitch[i] = pitch_detection(lowpass[shift:shift + frame_size], fs)
        shift += frame_rate

    frame_pitch = medfilt(frame_pitch, 5)

    pitch_contour = np.zeros(len_data)
    for i in range(0, len_data):
        pitch_contour[i] = frame_pitch[math.floor(i / frame_rate)]
    plt.plot(lowpass)
    plt.show()
    plt.plot(pitch_contour)
    plt.show()

    return pitch_contour


def pitch_detection(data, fs):
    min_lag = round(fs / 500)
    max_lag = round(fs / 70)

    #cc, _ = center_clipping(data, 0.3)
    cc = data - data.mean()
    auto_corr = np.correlate(cc, cc, mode='full')[-len(data):]
    auto_corr /= np.amax(auto_corr)
    #auto_corr /= data.var() * np.arange(len(data), 0, -1)
    auto_corr = auto_corr[max_lag + 1: 2 * max_lag]

    max_index = auto_corr[min_lag:max_lag].argmax()
    max_index += min_lag
    max_value = auto_corr[max_index]

    half_index = max_index // 2
    half_value = auto_corr[half_index]

    min_index = np.argmin(auto_corr[:max_index])
    min_value = auto_corr[min_index]

    mean_value = np.mean(auto_corr)

    if max_value > 0.35 and min_value < 0.0 and is_peak(
            max_index, min_lag, max_lag, auto_corr):
        pitch = fs / max_index
    else:
        pitch = 0.0
    return pitch


def center_clipping(data, percentage):
    max_amplitude = np.amax(np.abs(data))
    clip_level = max_amplitude * percentage
    positive_set = np.where(data > clip_level)
    negative_set = np.where(data < -clip_level)
    cc = np.zeros(len(data), dtype=np.float32)
    cc[positive_set] = data[positive_set] - clip_level
    cc[negative_set] = data[negative_set] + clip_level
    return cc, clip_level


def is_peak(i, min_lag, max_lag, data):
    if i == min_lag or i == max_lag:
        return False
    if data[i] < data[i - 1] or data[i] < data[i + 1]:
        return False
    return True


if __name__ == '__main__':
    main()
