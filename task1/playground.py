import math
import numpy as np
import scipy.signal as signal
import scipy.io as sio

from scipy.fftpack import fft, ifft


def noiseFFT(no, frameLen):
    flen = int(frameLen / 2)
    sqres = np.zeros(flen)
    mns = np.zeros(flen)
    k = 0
    for ind in range(0, len(no) - frameLen, frameLen):
        span = no[ind:ind + frameLen]
        fSpan = np.abs(fft(span))[0:flen]
        sqres += fSpan * fSpan
        mns += fSpan
        k += 1
    sqres /= k
    mns /= k
    return sqres - mns * mns


def restSign(sign, frameLen, sigNo, parm):
    ln = len(sign)
    flen = int(frameLen / 2)
    out = np.zeros((ln))
    for ind in range(0, ln - frameLen, frameLen):
        span = sign[ind:ind + frameLen]
        fSpan = fft(span)
        aFSpan = np.abs(fSpan)[0:flen]
        sqF = aFSpan * aFSpan - parm * sigNo
        nFour = np.complex_(np.zeros(frameLen))
        for lInd in range(flen):
            if sqF[lInd] > 0:
                nFour[lInd] = math.sqrt(sqF[lInd])
                nFour[lInd] *= fSpan[lInd] / aFSpan[lInd]
        for lInd in range(1, flen):
            nFour[frame - lInd] = np.conj(nFour[lInd])
        nFour[0] = 0
        inFour = ifft(nFour)
        out[ind:ind + frameLen] = np.real(inFour)
    return out


[fr, dt] = sio.wavfile.read('путь к зашумленному файлу')
[fr1, no] = sio.wavfile.read('путь к файлу, содержащий чистый шум')
frameLen = int(2 * round(fr / 200))

wnd = signal.hann(frameLen)
frame *= wnd
