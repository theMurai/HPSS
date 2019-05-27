import librosa
import numpy as np
import sys


def HPSS(x, p=2, harm_filter_lemgth=17, perc_filter_lemgth=17,
         is_soft_mask=True):
    X = librosa.core.stft(x, hop_length=441)
    X_abs = np.abs(X)
    F, T = X_abs.shape

    H = np.zeros((F, T))
    P = np.zeros((F, T))

    # 調波音のmedian filterを適用
    for t in range(T):
        init_frame = t - (harm_filter_lemgth - 1) // 2
        if (init_frame < 0):
            init_frame = 0

        end_frame = t + (harm_filter_lemgth - 1) // 2 + 1

        H[:, t] = np.median(X_abs[:, init_frame: end_frame], axis=1)

    # 非調波音のmedian filterを適用
    for f in range(F):
        init_frame = f - (perc_filter_lemgth - 1) // 2
        if (init_frame < 0):
            init_frame = 0

        end_frame = f + (perc_filter_lemgth - 1) // 2 + 1

        P[f, :] = np.median(X_abs[init_frame: end_frame, :], axis=0)

    if is_soft_mask:
        M_H = H ** p / (P ** p + H ** p + 1e-10)
        M_P = P ** p / (P ** p + H ** p + 1e-10)
    else:
        M_H = (H > P).astype(float)
        M_P = (H < P).astype(float)

    wav_Harm = librosa.core.istft(X * M_H, hop_length=441)
    wav_Perc = librosa.core.istft(X * M_P, hop_length=441)

    librosa.output.write_wav(
        "./Harm_FitzGerald2010.wav", wav_Harm, 44100)
    librosa.output.write_wav(
        "./Perc_FitzGerald2010.wav", wav_Perc, 44100)


def main(args):
    x, sr = librosa.core.load(args[1], sr=44100)
    HPSS(x)


if __name__ == "__main__":
    main(sys.argv)
