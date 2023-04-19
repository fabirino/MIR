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
    maxx = np.amax(features, axis=0)
    minn = np.amin(features, axis=0)
    normalized = (features - minn) / (maxx - minn)
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

    # print(bpm)


def extract_features(audio):
    mean = audio.mean()
    stdDev = audio.std()
    skewness = scs.skew(audio)
    kurtosis = scs.kurtosis(audio)
    median = np.median(audio)
    max_value = audio.max()
    min_value = audio.min()
    return np.array([mean, stdDev, skewness, kurtosis, median, max_value, min_value])


def features_array():
    array = np.empty((900, 7))
    index = 0
    for music in os.listdir("./MER_audio_dataset/audios"):
        path = "./MER_audio_dataset/audios/" + music
        y, sr = librosa.load(path)
        array[index] = extract_features(y)
        index +=1

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
    # np.savetxt("./Features/top100_feat_normalized.txt", normalized)

    # 2.2.1
    music_file = "./MER_audio_dataset/audios/MT0000004637.mp3"
    y, sr = librosa.load(music_file)
    # librosa_stats(y, sr)

    # 2.2.1
    extract_features(y)

    # 2.2.2
    all_features = features_array()

    # 2.2.3
    all_features_normalized = normalize_features(all_features)

    # 2.2.4
    # np.savetxt("./MER_audio_dataset/normalized_features.csv", all_features_normalized)


if __name__ == "__main__":
    Exercicio2()
