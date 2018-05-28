import numpy as np

from scipy.io import wavfile


def main():
    wav_file = 'male_ana'
    rate, data = wavfile.read('{}.wav'.format(wav_file))
    data_dtype = data.dtype
    data = pcm2fp(data)
    result = pitch_shift(data, 3)
    wavfile.write(
        '{}_filtered.wav'.format(wav_file), rate, fp2pcm(result, data_dtype))


def speedx(data, factor):
    indices = np.round(np.arange(0, len(data), factor))
    indices = indices[indices < len(data)].astype(np.int32)
    return data[indices]


def stretch(data, f, wnd_len, h):
    phase = np.zeros(wnd_len)
    wnd = np.hanning(wnd_len)
    result = np.zeros(int((len(data) + wnd_len) / f))

    for i in np.arange(0, len(data) - (wnd_len + h), h * f, dtype=np.int32):
        a1 = data[i:i + wnd_len]
        a2 = data[i + h:i + wnd_len + h]

        s1 = np.fft.fft(wnd * a1)
        s2 = np.fft.fft(wnd * a2)
        phase = (phase + np.angle(s2 / s1)) % 2 * np.pi

        a2_rephased = np.fft.ifft(np.abs(s2) * np.exp(1j * phase))
        i2 = int(i / f)
        result[i2:i2 + wnd_len] += wnd * np.real(a2_rephased)

    return result


def pitch_shift(data, n, wnd_len=2**12, h=2**6):
    factor = 2**(n / 12)
    stretched = stretch(data, 1.0 / factor, wnd_len, h)
    return speedx(stretched[wnd_len:], factor)


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


if __name__ == "__main__":
    main()
