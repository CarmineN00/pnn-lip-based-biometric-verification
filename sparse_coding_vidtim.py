
import cv2
from sklearn.decomposition import MiniBatchDictionaryLearning
from tqdm import tqdm
import csv
import os
import numpy as np
import glob
import read_data as rd
import pnn as p

def create_csv(csv_filename, directory, trainTestToggle):

    howmanyvideosfrom = {}
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        header_string = []

        for i in range(60000):
            header_string.append("Feature " + str(i))

        header_string.append("Label")
        writer.writerow(header_string)

        if os.path.isdir(directory):
            files = glob.glob(directory + "/**/*.avi", recursive=True)

            for video in tqdm(files, desc=directory, ncols=100):
                
                video_label = video.split("\\")[4].split("_")[0]
                if howmanyvideosfrom.get(video_label) is None:
                    howmanyvideosfrom[video_label] = 1
                else:
                    howmanyvideosfrom[video_label] = howmanyvideosfrom[video_label] + 1
                
                if trainTestToggle:
                    if howmanyvideosfrom[video_label]<=8:
                        res = get_features(video)
                        row = np.append(res,video_label)
                        writer.writerow(row)
                else:
                    if howmanyvideosfrom[video_label]>8:
                        res = get_features(video)
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

    #Uncomment these two lines to create the dataset, leave them commented if dataset already exists
    # Fare il download del dataset VidTim-Video-m prima ed inserirlo nella main directory del progetto prima di procedere

    train_csv_file = "vidtim_svd_train.csv"
    test_csv_file = "vidtim_svd_test.csv"

    if not os.path.isfile(train_csv_file):
        create_csv(train_csv_file,"VidTimit-Video-m",True)

    if not os.path.isfile(test_csv_file):
        create_csv(test_csv_file,"VidTimit-Video-m",False)

    data = rd.create_data_correctly(train_csv_file,test_csv_file)

    predictions = p.PNN(data, "rbf")

    evaluation_ready_ground_truth = np.array(data['y_test_before_ohe'], dtype=np.int64)

    scores = p.print_metrics(evaluation_ready_ground_truth, predictions)
    print("\nAccuracy: "+str(scores["accuracy"])+"\nPrecision: "+str(scores["precision"])+"\nRecall: "+str(scores["recall"])+"\n\n")