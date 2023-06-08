import cv2
from sklearn.decomposition import MiniBatchDictionaryLearning
from tqdm import tqdm
import csv
import os
import numpy as np
import glob
import read_data as rd
import pnn as p

# Carica il video di input

def create_csv(csv_filename, directory):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        header_string = []

        for i in range(60000):
            header_string.append("Feature " + str(i))

        header_string.append("Label")
        writer.writerow(header_string)

        if os.path.isdir(directory):
            files = glob.glob(directory + "/*.avi")

            for video in tqdm(files, desc=directory, ncols=100):
                res = get_features(video)
                video_label = str(''.join(video.split("\\")[1].split(".")[0].split("_")[:4]))
                row = np.append(res,video_label)
                writer.writerow(row)

def get_features(video_path):
    vidcap = cv2.VideoCapture(video_path)

    # Imposta i parametri per la codifica sparsa
    # N.B La spiegazione dei parametri settati è riportata nella prossima cella 
    n_components = 1
    '''alpha = 1
    batch_size = 3
    n_iter = 1000
    shuffle = True
    random_state = None
    positive_dict = False
    transform_algorithm = 'omp'
    transform_alpha = None
    transform_n_nonzero_coefs = None'''

    # Inizializza il modello di apprendimento del dizionario
    dict_learning = MiniBatchDictionaryLearning(n_components=n_components)

    # Loop attraverso i frame del video
    while True:
        # Leggi il frame successivo
        ret, frame = vidcap.read()
    
        # Se non ci sono più frame, esci dal loop
        if not ret:
            break
        
        # Converti l'immagine in scala di grigi
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Appiattisci l'immagine in un array 1D
        flat_frame = gray_frame.flatten()
        
        # Apprendi il dizionario sparsificato dal frame corrente
        dict_learning.partial_fit([flat_frame])

    # Recupera il dizionario appreso
    dictionary = dict_learning.components_[0]

    return dictionary


if __name__ == '__main__':

    #Uncomment these lines to create the dataset, leave them commented if dataset already exists

    create_csv("svd_test.csv","Lips/Test")
    create_csv("svd_train.csv","Lips/Train")
    data = rd.create_data_correctly("svd_train.csv","svd_test.csv")
    predictions = p.PNN(data, "rbf")
    scores = p.print_metrics(data['y_test_before_ohe'], predictions)
    print("\nAccuracy: "+str(scores["accuracy"])+"\nPrecision: "+str(scores["precision"])+"\nRecall: "+str(scores["recall"])+"\n\n")
    