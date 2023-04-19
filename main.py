import numpy as np
import csv
from io import StringIO

# PASTAS 
# "./MER_audio_dataset/audios"
# "./Queries"

def array_features(path):
    features = np.genfromtxt(path, delimiter=",", dtype=str)
    features = np.delete(features, 0, 0)  # eliminar a primeira linha
    features = np.delete(features, 0 ,1)  # eliminar a primeira coluna
    features = np.delete(features, -1 ,1) # eliminar a primeira coluna
    return features.astype(float)

def normalize_features(features):
    max = np.amax(features, axis=0)
    min = np.amin(features, axis=0)
    normalized = (features - min) / (max - min)
    return normalized

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

    # 2.2.2



if __name__ == "__main__":
    Exercicio2()
