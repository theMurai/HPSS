import librosa
import numpy as np
import sys
eps = 1e-10


def HPSS(x, harm_filter_lemgth=17, perc_filter_lemgth=17, beta_harm=2,
         beta_perc=2):
    X_first = librosa.core.stft(x)
    X_fitst_abs = np.abs(X_first)
    F, T = X_fitst_abs.shape

    H = np.zeros((F, T))
    P = np.zeros((F, T))

    # 調波音のmedian filterを適用
    for t in range(T):
        init_frame = max([t - (harm_filter_lemgth - 1) // 2, 0])
        end_frame = t + (harm_filter_lemgth - 1) // 2 + 1

        H[:, t] = np.median(X_fitst_abs[:, init_frame: end_frame], axis=1)

    # 非調波音のmedian filterを適用
    for f in range(F):
        init_frame = max([f - (perc_filter_lemgth - 1) // 2, 0])
        end_frame = f + (perc_filter_lemgth - 1) // 2 + 1

        P[f, :] = np.median(X_fitst_abs[init_frame: end_frame, :], axis=0)

    M = ((H / (P + eps)) > beta_harm).astype(float)
    X_harm = X_first * M

    x_second = librosa.core.istft(X_first - X_harm)
    X_second = librosa.core.stft(x_second)

    X_second_abs = np.abs(X_second)
    F, T = X_second_abs.shape

    H = np.zeros((F, T))
    P = np.zeros((F, T))

    # 調波音のmedian filterを適用
    for t in range(T):
        init_frame = t - (harm_filter_lemgth - 1) // 2
        if (init_frame < 0):
            init_frame = 0

        end_frame = t + (harm_filter_lemgth - 1) // 2 + 1

        H[:, t] = np.median(X_second_abs[:, init_frame: end_frame], axis=1)

    # 非調波音のmedian filterを適用
    for f in range(F):
        init_frame = f - (perc_filter_lemgth - 1) // 2
        if (init_frame < 0):
            init_frame = 0

        end_frame = f + (perc_filter_lemgth - 1) // 2 + 1

        P[f, :] = np.median(X_second_abs[init_frame: end_frame, :], axis=0)

    M = ((P / (H + eps)) > beta_perc).astype(float)
    X_perc = X_second * M
    X_remain = X_second * (1 - M)

    wav_Harm = librosa.core.istft(X_harm)
    wav_Perc = librosa.core.istft(X_perc)
    wav_Remain = librosa.core.istft(X_remain)

    librosa.output.write_wav(
        "./Harm_Driedger2014.wav", wav_Harm, 44100)
    librosa.output.write_wav(
        "./Perc_Driedger2014.wav", wav_Perc, 44100)
    librosa.output.write_wav(
        "./Remain_Driedger2014.wav", wav_Remain, 44100)


def main(args):
    x, sr = librosa.core.load(args[1], sr=44100)
    HPSS(x)


if __name__ == "__main__":
    main(sys.argv)
