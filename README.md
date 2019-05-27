HPSS using Median-filter
========

メディアンフィルターを用いた調波非調波音分離(HPSS: Harmonic/Percussive Source Separation)を2種類実装しました。


## Description
Pythonで2種類のHPSSを実装しました。
1つ目はFitzGerald(2010)による、メディアンフィルターを使うHPSSです。
2つ目はDriedgerら(2014)による、1つ目のHPSSに改良を加えたものです。

この2種類のHPSSをもとに、音声信号処理ライブラリlibrosaのlibrosa.decompose.HPSSが実装されています。

## 言語・環境
言語 　　　: Python3
ライブラリ : librosa、numpy

## 使用方法
python3 FitzGerald_2010.py [input_wav_file(44100Hz)]
python3 Driedger_2014.py [input_wav_file(44100Hz)]

## 参考文献
Fitzgerald, Derry. “Harmonic/percussive separation using median filtering.” 13th International Conference on Digital Audio Effects (DAFX10), 2010.
Driedger, Müller, Disch. “Extending harmonic-percussive separation of audio.” 15th International Society for Music Information Retrieval Conference (ISMIR 2014), 2014.