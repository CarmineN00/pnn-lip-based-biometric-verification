import math
import numpy as np
import os, glob, cv2
import mediapipe as mp
import pandas
import LipLandmarks as lp
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from tqdm import tqdm

num_frames = 3


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


def create_data(train_csv_filename, test_csv_filename):
    # Creazione dataframe di training

    df_train = pandas.read_csv(train_csv_filename)
    df_train.sort_values(by=['Label'])

    x_train = df_train.iloc[:, :-1].values
    y_train = df_train['Label'].values

    j = 0
    last_label_train = y_train[0]
    y_train[0] = j

    for i in range(1, len(y_train)):
        if y_train[i] == last_label_train:
            y_train[i] = j
        else:
            last_label_train = y_train[i]
            j = j + 1
            y_train[i] = j

    x_train = scale(x_train)

    # Creazione dataframe di test

    df_test = pandas.read_csv(test_csv_filename)
    df_test.sort_values(by=['Label'])

    x_test = df_test.iloc[:, :-1].values
    y_test = df_test['Label'].values

    j = 0
    last_label = y_test[0]
    y_test[0] = j

    for i in range(1, len(y_test)):
        if y_test[i] == last_label:
            y_test[i] = j
        else:
            last_label = y_test[i]
            j = j + 1
            y_test[i] = j

    x_test = scale(x_test)

    # Dictionary che contiene i dati di training e test

    data = {
        'x_train': x_train,
        'x_test': x_test,
        'y_train': ohe(y_train),
        'y_test': ohe(y_test)
    }

    return data


def ohe(y):
    y_ohe = np.zeros((y.shape[0], np.unique(y).shape[0]))
    
    for i in range(y.shape[0]):
        y_ohe[i, y[i]] = 1

    pandas.DataFrame(y_ohe).to_csv("ohe_classes.csv")

    return y_ohe


if __name__ == "__main__":
    create_csv("test_dataset.csv")
    create_csv("train_dataset.csv")
