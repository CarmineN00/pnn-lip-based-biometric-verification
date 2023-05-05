import math
import numpy as np
import os, glob, cv2
import mediapipe as mp
import pandas
import LipLandmarks as lp
import csv
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

from tqdm import tqdm


def ottieni_lista_distanze_euclidee(filename):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    cap = cv2.VideoCapture(filename)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 4)

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


def create_csv(csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        header_string = []
        for i in range(20):
            header_string.append("Feature "+str(i))
        header_string.append("Label")
        #writer.writerow(header_string)

        os.chdir("Dataset")
        video_dir = os.listdir()

        for directory in video_dir:
            if os.path.isdir(directory):
                files = glob.glob(directory + "/*.avi")

                # Progress bar
                for video in tqdm(files, desc=directory, ncols=100):
                    res = ottieni_lista_distanze_euclidee(video)
                    # Se necessario, cambiare qui il metodo di prelievo della label
                    video_label = str(''.join(video.split("\\")[1].split(".")[0].split("_")[:4]))
                    if np.shape(res)[0] == 60:
                        res_split = np.array_split(res, 3)
                        for i in range(3):
                            info_row = res_split[i]
                            writer.writerow(np.append(info_row, video_label)) 


def create_df(csv_filename):
    df = pandas.read_csv(csv_filename)
    df.sort_values(by=['Label'])

    x = df.iloc[:, :-1].values
    y = df['Label'].values

    j = 1
    last_label = y[0]
    y[0] = j
    #print(last_label, "->", y[0])

    for i in range(1, len(y)):
        to_print = str(y[i])
        if y[i] == last_label:
            y[i] = j
        else:
            last_label = y[i]
            j = j+1
            y[i] = j
        #print(to_print, "->", y[i])
        
    #print(f'Shape X: {np.shape(x)}, Shape Y: {np.shape(y)}')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

    data = {'x_train': x_train, 
			'x_test': x_test, 
			'y_train': y_train, 
			'y_test': y_test}

    return data


if __name__ == "__main__":
    create_csv()
