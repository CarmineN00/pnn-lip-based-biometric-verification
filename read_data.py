import pandas
import numpy as np
import os, glob, cv2
import math
import mediapipe as mp
import pandas
import LipLandmarks as lp
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from tqdm import tqdm

num_frames = 20

def create_hashmap_from_ugly_labels_to_numbers(list):
    hashmap = {}
    i = 0
    for elem in list:
        if hashmap.get(elem) is None:
            hashmap[elem] = i
            i = i+1
    return hashmap

def ohe_correct(y, width):
    y_ohe = np.zeros((y.shape[0], width))
    
    for i in range(y.shape[0]):
        y_ohe[i, y[i]] = 1

    pandas.DataFrame(y_ohe).to_csv("ohe_classes.csv")

    return y_ohe

def old_get_labels(directories):
    video_label_list = []
    for directory in directories:
        if os.path.isdir(directory):
            files = glob.glob(directory + "/*.avi")
            for video in files:
               video_label = str(''.join(video.split("\\")[1].split(".")[0].split("_")[:4]))
               video_label_list.append(video_label)
    return video_label_list

def get_labels(csvs):
    video_labels = []
    for csv in csvs:
        df_train = pandas.read_csv(csv)
        y_train = df_train['Label'].values
        for elem in y_train:
            video_labels.append(elem)
    return video_labels

def create_data_correctly(train_csv_filename, test_csv_filename):
    #Ottengo tutte le possibili label, anche con ripetizioni
    video_label_list = get_labels(["test_dataset.csv","train_dataset.csv"])

    #Determino quante distinte persone ci sono in tutte le cartelle
    num_distinct_people = len(np.unique(video_label_list))
    print("\nNumero reali individui distinti: ",num_distinct_people)

    #Creo una hashmap che associa ad ogni label vecchia un numero da 0 a 255
    hashmap = create_hashmap_from_ugly_labels_to_numbers(video_label_list)
    #print("\nHashMap\n", hashmap)

    #Prendo il dataset train e converto le y dalle label vecchie ad rispettivo id da 0 a 255
    df_train = pandas.read_csv("train_dataset.csv")
    x_train = df_train.iloc[:, :-1].values
    y_train = df_train['Label'].values

    #print("Before converting y_train: ", y_train)

    for j in range (len(y_train)):
        #print(y_train[j], "->", hashmap[str(y_train[j])])
        y_train[j] = hashmap[y_train[j]]

    #print("After converting y_train: ", y_train)

    #Ora, la stessa conversione anche per il dataset test

    df_test = pandas.read_csv("test_dataset.csv")
    df_test.sort_values(by=['Label'])

    x_test = df_test.iloc[:, :-1].values
    y_test = df_test['Label'].values

    #print("Before converting y_test: ", y_test)

    for j in range (len(y_test)):
        #print(y_test[j], "->", hashmap[str(y_test[j])])
        y_test[j] = hashmap[y_test[j]]

    #print("After converting y_test: ", y_test)

    # Ora e soltanto ora possiamo usare il one-hot-encoding!

    data = {
        'x_train': scale(x_train),
        'x_test': scale(x_test),
        'y_train': ohe_correct(y_train, num_distinct_people),
        'y_test': ohe_correct(y_test, num_distinct_people),
        #Nel caso dovesse servire, fornisco anche la hashmap
        'hashmap': hashmap,
        'y_test_before_ohe': y_test,
        'y_train_before_ohe': y_train
    }

    '''print("Shape of data['x_train']:", np.shape(data['x_train']))
    print("Shape of data['x_test']:", np.shape(data['x_test']))
    print("Shape of data['y_train']:", np.shape(data['y_train']))
    print("Shape of data['y_test']:", np.shape(data['y_test']))'''

    return data


def ottieni_lista_distanze_euclidee(filename):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    cap = cv2.VideoCapture(filename)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - num_frames - 1)

    ret, frame = cap.read()

    landmarks = []

    list_of_euclidean_distances = []

    while ret:

        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb_frame)

        iterator = iter(lp.lip_landmarks)
        for j in range(0, len(lp.lip_landmarks)):
            i = next(iterator)

            '''
                Non è detto che da un video sia sempre possibile 
                fare detection di almeno un volto, quindi questo 
                controllo è strettamente necessario
            '''
            if results and results.multi_face_landmarks:
                # Primo elemento della tupla
                point = results.multi_face_landmarks[0].landmark[i[0]]
                node1_x = int(point.x * width)
                node1_y = int(point.y * height)
                landmarks.append((node1_x, node1_y))
                cv2.circle(frame, (node1_x, node1_y), 2, (255, 255, 255), -1)

                # Secondo elemento della tupla
                point = results.multi_face_landmarks[0].landmark[i[1]]
                node2_x = int(point.x * width)
                node2_y = int(point.y * height)
                landmarks.append((node2_x, node2_y))
                cv2.circle(frame, (node2_x, node2_y), 2, (255, 255, 255), -1)

                # Calcolo della distanza euclidea tra i punti
                d = math.sqrt((node2_x - node1_x) ** 2 + (node2_y - node1_y) ** 2)
                list_of_euclidean_distances.append(d)

    return list_of_euclidean_distances

def create_csv(csv_filename, directory):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        header_string = []

        for i in range(20):
            header_string.append("Feature " + str(i))

        header_string.append("Label")
        writer.writerow(header_string)

        if os.path.isdir(directory):
            files = glob.glob(directory + "/*.avi")

            for video in tqdm(files, desc=directory, ncols=100):
                res = ottieni_lista_distanze_euclidee(video)
                video_label = str(''.join(video.split("\\")[1].split(".")[0].split("_")[:4]))

                if np.shape(res)[0] == 20 * num_frames:
                    res_split = np.array_split(res, num_frames)

                    for i in range(num_frames):
                        info_row = res_split[i]
                        writer.writerow(np.append(info_row, video_label))
                '''else:
                    print("Failed to fetch lip features from video: ",video)'''

if __name__ == "__main__":
    '''create_csv("test_dataset.csv", "Dataset/Test")
    create_csv("train_dataset.csv", "Dataset/Train")'''
    create_data_correctly("train_dataset.csv", "test_dataset.csv")

