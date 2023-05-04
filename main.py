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
    f0[f0 == 11025] = 0
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
        f0_aux = librosa.yin(y, fmin=20, fmax=11025)
        f0_aux[f0_aux == 11025] = 0
        f0 = np.apply_along_axis(
            extract_features, 0, f0_aux)
        rms = np.apply_along_axis(
            extract_features, 1, librosa.feature.rms(y=y)).flatten()
        zcr = np.apply_along_axis(
            extract_features, 1, librosa.feature.zero_crossing_rate(y=y)).flatten()

        bpm = librosa.beat.tempo(y=y)

        aux = np.concatenate((mfcc, centroid, bswth, contrast, flatness, rollof,
                              f0, rms, zcr, bpm)).astype(float)

        # print(aux)
        array[index] = aux
        index += 1

    # Substituir os valores nan por 0
    nan_mask = np.isnan(array)
    array[nan_mask] = np.nan_to_num(array[nan_mask], nan=0)

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
    np.savetxt("./Features/top100_feat_normalized.csv",
               normalized, delimiter=";", fmt='%f')

    # 2.2.1
    music_file = "./MER_audio_dataset/audios/MT0000004637.mp3"
    y, sr = librosa.load(music_file)
    # librosa_stats(y, sr)

    # 2.2.1
    # extract_features(y)

    # 2.2.2
    all_features = all_features_array()  # para todas as features

    # save features
    np.savetxt("./MER_audio_dataset/not_Norm_features.csv",
               all_features, delimiter=",", fmt='%f')

    # 2.2.3
    all_features_normalized = normalize_features(all_features)

    # 2.2.4
    np.savetxt("./MER_audio_dataset/normalized_features.csv",
               all_features_normalized, delimiter=";", fmt='%f')

# ==================================================================================================


def euclidean_distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))


def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def E3_2(matrix, name):

    similarity_matrix = np.zeros((900, 900))
    for i in range(900):
        for j in range(i+1, 900):
            d = euclidean_distance(matrix[i, :], matrix[j, :])
            similarity_matrix[i, j] = d
            similarity_matrix[j, i] = d
    np.savetxt(
        f"./MER_audio_dataset/SimilarityMatrix/{name}_euclidean.csv", similarity_matrix, delimiter=";", fmt='%f')

    similarity_matrix = np.zeros((900, 900))
    for i in range(900):
        for j in range(i+1, 900):
            d = manhattan_distance(matrix[i, :], matrix[j, :])
            similarity_matrix[i, j] = d
            similarity_matrix[j, i] = d
    np.savetxt(
        f"./MER_audio_dataset/SimilarityMatrix/{name}_manhattan.csv", similarity_matrix, delimiter=";", fmt='%f')

    similarity_matrix = np.zeros((900, 900))
    for i in range(900):
        for j in range(i+1, 900):
            d = cosine_distance(matrix[i, :], matrix[j, :])
            similarity_matrix[i, j] = d
            similarity_matrix[j, i] = d
    np.savetxt(
        f"./MER_audio_dataset/SimilarityMatrix/{name}_cosine.csv", similarity_matrix, delimiter=";", fmt='%f')


def get_song_index(song_name):
    songs = os.listdir("./MER_audio_dataset/audios")
    songs.sort()
    return np.where(np.array(songs) == song_name)[0][0]


def get_song_name(song_index):
    songs = os.listdir("./MER_audio_dataset/audios")
    songs.sort()
    return songs[song_index]


def rank_songs(name):
    for song in os.listdir("./MER_audio_dataset/audios"):
        i = get_song_index(song)
        song_row = np.loadtxt(
            f"./MER_audio_dataset/SimilarityMatrix/{name}_euclidean.csv", delimiter=";")[i]
        sorted_indexes = np.argsort(song_row)[0:21]
        top_songs_names = list(
            map(lambda song_index: get_song_name(song_index), sorted_indexes))
        os.makedirs(f"./MER_audio_dataset/Rankings/{song}", exist_ok=True)
        np.savetxt(
            f"./MER_audio_dataset/Rankings/{song}/{name}_euclidean.csv", top_songs_names, fmt="%s")

        song_row = np.loadtxt(
            f"./MER_audio_dataset/SimilarityMatrix/{name}_manhattan.csv", delimiter=";")[i]
        sorted_indexes = np.argsort(song_row)[0:21]
        top_songs_names = list(
            map(lambda song_index: get_song_name(song_index), sorted_indexes))
        os.makedirs(f"./MER_audio_dataset/Rankings/{song}", exist_ok=True)
        np.savetxt(
            f"./MER_audio_dataset/Rankings/{song}/{name}_manhattan.csv", top_songs_names, fmt="%s")

        song_row = np.loadtxt(
            f"./MER_audio_dataset/SimilarityMatrix/{name}_cosine.csv", delimiter=";")[i]
        sorted_indexes = np.argsort(song_row)[0:21]
        top_songs_names = list(
            map(lambda song_index: get_song_name(song_index), sorted_indexes))
        os.makedirs(f"./MER_audio_dataset/Rankings/{song}", exist_ok=True)
        np.savetxt(
            f"./MER_audio_dataset/Rankings/{song}/{name}_cosine.csv", top_songs_names, fmt="%s")


def Exercicio3():
    all_features_normalized = np.loadtxt(
        "./MER_audio_dataset/normalized_features.csv", delimiter=";")
    top100 = np.loadtxt("./Features/top100_feat_normalized.csv", delimiter=";")
    os.makedirs(f"./MER_audio_dataset/SimilarityMatrix", exist_ok=True)

    E3_2(top100, "top100")
    E3_2(all_features_normalized, "librosa")

    os.makedirs(f"./MER_audio_dataset/Rankings", exist_ok=True)
    rank_songs("librosa")
    rank_songs("top100")

# ==================================================================================================


def create_sim_table(metadados):
    table = np.zeros((len(metadados), len(metadados)))
    for i in range(len(metadados)):
        mood1 = [i.strip("\" ") for i in metadados[i][3].split(";")]
        gen1 = [i.strip("\" ") for i in metadados[i][4].split(";")]

        for j in range(i, len(metadados)):
            count = 0

            # Artist
            if metadados[i][1] == metadados[j][1]:
                count += 1

            # Quadrant
            if metadados[i][2] == metadados[j][2]:
                count += 1

            # MoodsStrSplit
            mood2 = [i.strip("\" ") for i in metadados[j][3].split(";")]
            matches = np.intersect1d(mood1, mood2)
            count += len(matches)

            # GenreStr
            gen2 = [i.strip("\" ") for i in metadados[j][4].split(";")]
            matches2 = np.intersect1d(gen1, gen2)
            count += len(matches2)

            table[i][j] = count
            table[j][i] = count

    return table


def top20_musics(metadados, sim_table):
    q1_cosine = np.loadtxt(
        "./MER_audio_dataset/Rankings/MT0000202045.mp3/librosa_cosine.csv", dtype=str)
    q1_cosine = np.char.strip(q1_cosine, "\" ")
    q1_euclidean = np.loadtxt(
        "./MER_audio_dataset/Rankings/MT0000202045.mp3/librosa_euclidean.csv", dtype=str)
    q1_euclidean = np.char.strip(q1_euclidean, "\" ")
    q1_manhattan = np.loadtxt(
        "./MER_audio_dataset/Rankings/MT0000202045.mp3/librosa_manhattan.csv", dtype=str)
    q1_manhattan = np.char.strip(q1_manhattan, "\" ")

    q2_cosine = np.loadtxt(
        "./MER_audio_dataset/Rankings/MT0000379144.mp3/librosa_cosine.csv", dtype=str)
    q2_cosine = np.char.strip(q2_cosine, "\" ")
    q2_euclidean = np.loadtxt(
        "./MER_audio_dataset/Rankings/MT0000379144.mp3/librosa_euclidean.csv", dtype=str)
    q2_euclidean = np.char.strip(q2_euclidean, "\" ")
    q2_manhattan = np.loadtxt(
        "./MER_audio_dataset/Rankings/MT0000379144.mp3/librosa_manhattan.csv", dtype=str)
    q2_manhattan = np.char.strip(q2_manhattan, "\" ")

    q3_cosine = np.loadtxt(
        "./MER_audio_dataset/Rankings/MT0000414517.mp3/librosa_cosine.csv", dtype=str)
    q3_cosine = np.char.strip(q3_cosine, "\" ")
    q3_euclidean = np.loadtxt(
        "./MER_audio_dataset/Rankings/MT0000414517.mp3/librosa_euclidean.csv", dtype=str)
    q3_euclidean = np.char.strip(q3_euclidean, "\" ")
    q3_manhattan = np.loadtxt(
        "./MER_audio_dataset/Rankings/MT0000414517.mp3/librosa_manhattan.csv", dtype=str)
    q3_manhattan = np.char.strip(q3_manhattan, "\" ")

    q4_cosine = np.loadtxt(
        "./MER_audio_dataset/Rankings/MT0000956340.mp3/librosa_cosine.csv", dtype=str)
    q4_cosine = np.char.strip(q4_cosine, "\" ")
    q4_euclidean = np.loadtxt(
        "./MER_audio_dataset/Rankings/MT0000956340.mp3/librosa_euclidean.csv", dtype=str)
    q4_euclidean = np.char.strip(q4_euclidean, "\" ")
    q4_manhattan = np.loadtxt(
        "./MER_audio_dataset/Rankings/MT0000956340.mp3/librosa_manhattan.csv", dtype=str)
    q4_manhattan = np.char.strip(q4_manhattan, "\" ")

    for music in os.listdir("./Queries"):

        index = np.where(music[:-4] == metadados[:, 0])[0][0]
        row = sim_table[index]
        sorted_row = np.argsort(row)[::-1]
        top20 = sorted_row[:21]
        top20_musics = metadados[top20][:, 0]
        top20_musics = np.char.add(top20_musics, ".mp3")
        song = metadados[index][0].strip("\" ")
        score = row[top20]

        lib_euclidean = np.loadtxt(
            f"./MER_audio_dataset/Rankings/{song}.mp3/librosa_euclidean.csv", dtype=str)
        lib_euclidean = np.char.strip(lib_euclidean, "\" ")

        der = np.array([])
        dmr = np.array([])
        dcr = np.array([])

        der = np.append(der, len(np.intersect1d(top20_musics, q1_euclidean)))
        der = np.append(der, len(np.intersect1d(top20_musics, q2_euclidean)))
        der = np.append(der, len(np.intersect1d(top20_musics, q3_euclidean)))
        der = np.append(der, len(np.intersect1d(top20_musics, q4_euclidean)))

        dmr = np.append(dmr, len(np.intersect1d(top20_musics, q1_manhattan)))
        dmr = np.append(dmr, len(np.intersect1d(top20_musics, q2_manhattan)))
        dmr = np.append(dmr, len(np.intersect1d(top20_musics, q3_manhattan)))
        dmr = np.append(dmr, len(np.intersect1d(top20_musics, q4_manhattan)))

        dcr = np.append(dcr, len(np.intersect1d(top20_musics, q1_cosine)))
        dcr = np.append(dcr,len(np.intersect1d(top20_musics, q2_cosine)))
        dcr = np.append(dcr,len(np.intersect1d(top20_musics, q3_cosine)))
        dcr = np.append(dcr,len(np.intersect1d(top20_musics, q4_cosine)))

        mean_der = np.sum(der)/len(der)
        mean_dmr = np.sum(dmr)/len(dmr)
        mean_dcr = np.sum(dcr)/len(dcr)

        with open(f"./MER_audio_dataset/Rankings/ranking{song}.txt", "w") as f:
            f.write(f"Query = '{song}.mp3'\n\n")
            f.write(
                f"Ranking = Ranking: FMrosa, Euclidean-------------\n{lib_euclidean}\n\n")
            f.write(
                f"Ranking = Ranking: Metadata-------------\n{top20_musics}\n\n")
            f.write(f"Score Metadata = {score}\n\n\n")
            f.write(f"Precision der: {der} *** {mean_der}\n")
            f.write(f"Precision dmr: {dmr} *** {mean_dmr}\n")
            f.write(f"Precision dcr: {dcr} *** {mean_dcr}\n")


def Exercicio4():
    metadados = np.loadtxt(
        "./MER_audio_dataset/panda_dataset_taffc_metadata.csv", dtype=str, delimiter=",")
    metadados = metadados[:, [0, 1, 3, 9, 11]]
    metadados = metadados[1:, :]
    metadados = np.char.strip(metadados, "\" ")

    # Achei mais facil fazer o segundo exercicio primeiro que assim ja ficamos com a tabela
    # 4.1.2
    # sim_table = create_sim_table(metadados)
    sim_table = np.loadtxt(
        "./MER_audio_dataset/SimilarityMatrix/sim_table.csv", delimiter=";")
    # np.savetxt("./MER_audio_dataset/sim_table.csv", sim_table, delimiter=";", fmt='%d')
    # 4.1.1
    top20_musics(metadados, sim_table)


if __name__ == "__main__":
    Exercicio4()
