import csv
import os
import warnings
from io import StringIO
import librosa
import numpy as np
import scipy.stats as scs

# PASTAS
# "./MER_audio_dataset/audios"
# "./Queries"

warnings.filterwarnings("ignore")


def array_features(path):
    features = np.genfromtxt(path, delimiter=",", dtype=str)
    features = np.delete(features, 0, 0)  # eliminar a primeira linha
    features = np.delete(features, 0, 1)  # eliminar a primeira coluna
    features = np.delete(features, -1, 1)  # eliminar a primeira coluna
    return features.astype(float)


def normalize_features(features):
    np.set_printoptions(suppress=True, precision=6)
    maxx = np.amax(features, axis=0)
    minn = np.amin(features, axis=0)
    normalized = (features - minn) / (maxx - minn)
    normalized = normalized.astype(float)

    # Substituir os valores nan por 0
    nan_mask = np.isnan(normalized)
    normalized[nan_mask] = np.nan_to_num(normalized[nan_mask], nan=0)

    return normalized


def librosa_stats(y, sr):
    # Espectrais
    mfcc = librosa.feature.mfcc(y=y, n_mfcc=13).flatten()
    centroid = librosa.feature.spectral_centroid(y=y).flatten()
    bswth = librosa.feature.spectral_bandwidth(y=y).flatten()
    contrast = librosa.feature.spectral_contrast(y=y).flatten()
    flatness = librosa.feature.spectral_flatness(y=y).flatten()
    rolloff = librosa.feature.spectral_rolloff(y=y).flatten()

    # Temporais
    f0 = librosa.yin(y, fmin=20, fmax=11025)
    rms = librosa.feature.rms(y=y).flatten()
    zcr = librosa.feature.zero_crossing_rate(y=y).flatten()

    bpm = librosa.beat.tempo(y=y)

    print("MFCC: ", mfcc, "\n", "Centroid: ", centroid, "\n", "Bandwidth: ", bswth, "\n", "Contrast: ", contrast, "\n", "Flatness: ",
          flatness, "\n", "Rolloff: ", rolloff, "\n", "F0: ", f0, "\n", "Rms: ", rms, "\n", "Zcr: ", zcr, "\n", "Bpm: ", bpm)


def extract_features(audio):
    mean = audio.mean()
    stdDev = audio.std()
    skewness = scs.skew(audio)
    kurtosis = scs.kurtosis(audio)
    median = np.median(audio)
    max_value = audio.max()
    min_value = audio.min()
    aux = np.array([mean, stdDev, skewness, kurtosis,
                   median, max_value, min_value])

    # Substituir os valores nan por 0
    nan_mask = np.isnan(aux)
    aux[nan_mask] = np.nan_to_num(aux[nan_mask], nan=0)

    return aux


def features_array():
    array = np.empty((900, 7))
    index = 0
    for music in os.listdir("./MER_audio_dataset/audios"):
        path = "./MER_audio_dataset/audios/" + music
        y, sr = librosa.load(path)
        array[index] = extract_features(y)
        index += 1

    return array


def all_features_array():
    array = np.empty((900, 190))  # 900 musicas, 190 features
    index = 0
    for music in os.listdir("./MER_audio_dataset/audios"):
        path = "./MER_audio_dataset/audios/" + music
        y, sr = librosa.load(path)

        # Espectrais
        mfcc = np.apply_along_axis(
            extract_features, 1, librosa.feature.mfcc(y=y, n_mfcc=13)).flatten()
        centroid = np.apply_along_axis(
            extract_features, 1, librosa.feature.spectral_centroid(y=y)).flatten()
        bswth = np.apply_along_axis(
            extract_features, 1, librosa.feature.spectral_bandwidth(y=y)).flatten()
        contrast = np.apply_along_axis(
            extract_features, 1, librosa.feature.spectral_contrast(y=y)).flatten()
        flatness = np.apply_along_axis(
            extract_features, 1, librosa.feature.spectral_flatness(y=y)).flatten()
        rollof = np.apply_along_axis(
            extract_features, 1, librosa.feature.spectral_rolloff(y=y)).flatten()

        # Temporais
        f0 = np.apply_along_axis(
            extract_features, 0, librosa.yin(y, fmin=20, fmax=11025))
        rms = np.apply_along_axis(
            extract_features, 1, librosa.feature.rms(y=y)).flatten()
        zcr = np.apply_along_axis(
            extract_features, 1, librosa.feature.zero_crossing_rate(y=y)).flatten()

        bpm = librosa.beat.tempo(y=y)

        aux = np.concatenate((mfcc, centroid, bswth, contrast, flatness, rollof,
                              f0, rms, zcr, bpm)).astype(float)

        nan_mask = np.isnan(aux)

        # Substituir os valores nan por 0
        aux[nan_mask] = np.nan_to_num(aux[nan_mask], nan=0)

        # print(aux)
        array[index] = aux
        index += 1

    # print(array)
    return array


def Exercicio2():
    # 2.1.1
    file_name = "./Features/top100_features.csv"
    features = array_features(file_name)
    # print(features)

    # 2.1.2
    normalized = normalize_features(features)
    # print(normalized)

    # 2.1.3
    np.savetxt("./Features/top100_feat_normalized.csv", normalized,delimiter=",", fmt='%.6f')

    # 2.2.1
    # music_file = "./MER_audio_dataset/audios/MT0000004637.mp3"
    # y, sr = librosa.load(music_file)
    # librosa_stats(y, sr)

    # 2.2.1
    # extract_features(y)

    # 2.2.2
    # all_features = features_array()
    # all_features = all_features_array()  # para todas as features

    #save features
    # np.savetxt("./MER_audio_dataset/not_Norm_features.csv",
            #    all_features, delimiter=",", fmt='%.6f')

    # 2.2.3
    # all_features_normalized = normalize_features(all_features)

    # 2.2.4
    # np.savetxt("./MER_audio_dataset/normalized_features.csv",
            #    all_features_normalized, delimiter=";", fmt='%.6f')


if __name__ == "__main__":
    Exercicio2()
