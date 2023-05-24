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
from scipy.spatial import Delaunay

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
    video_label_list = get_labels([train_csv_filename,test_csv_filename])

    #print("Video label list: ",video_label_list)

    #Determino quante distinte persone ci sono in tutte le cartelle
    #print("Video Label List Unique: ",np.unique(video_label_list))
    num_distinct_people = len(np.unique(video_label_list))
    #print("\nNumero reali individui distinti: ",num_distinct_people)

    #Creo una hashmap che associa ad ogni label vecchia un numero da 0 a 255
    hashmap = create_hashmap_from_ugly_labels_to_numbers(video_label_list)
    #print("\nHashMap\n", hashmap)

    #Prendo il dataset train e converto le y dalle label vecchie ad rispettivo id da 0 a 255
    df_train = pandas.read_csv(train_csv_filename)
    x_train = df_train.iloc[:, :-1].values
    y_train = df_train['Label'].values

    #print("Before converting y_train: ", y_train)

    for j in range (len(y_train)):
        #print(y_train[j], "->", hashmap[str(y_train[j])])
        y_train[j] = hashmap[y_train[j]]

    #print("After converting y_train: ", y_train)

    #Ora, la stessa conversione anche per il dataset test

    df_test = pandas.read_csv(test_csv_filename)
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
        'x_train': x_train,
        'x_test': x_test,
        'y_train': ohe_correct(y_train, num_distinct_people),
        'y_test': ohe_correct(y_test, num_distinct_people),
        #Nel caso dovesse servire, fornisco anche la hashmap
        'hashmap': hashmap,
        'y_test_before_ohe': y_test,
        'y_train_before_ohe': y_train
    }

    return data

def ottieni_features_da_video(filename,type, num_frames):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    cap = cv2.VideoCapture(filename)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - num_frames - 1)

    ret, frame = cap.read()

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

            if results and results.multi_face_landmarks:
                # Primo elemento della tupla
                point1 = results.multi_face_landmarks[0].landmark[i[0]]
                node1_x = int(point1.x * width)
                node1_y = int(point1.y * height)

                # Secondo elemento della tupla
                point2 = results.multi_face_landmarks[0].landmark[i[1]]
                node2_x = int(point2.x * width)
                node2_y = int(point2.y * height)

                # Calcolo della distanza euclidea tra i punti
                if (type == "DistanzeEuclidee2D"):
                    d = math.sqrt((node2_x - node1_x) ** 2 + (node2_y - node1_y) ** 2)
                elif (type == "DistanzeEuclideeNormalizzate2D"):
                    d = math.sqrt((point2.x - point1.y) ** 2 + (point2.y - point1.y) ** 2)
                elif (type == "CityBlock3D"):
                    d = abs(point1.x - point2.x) + abs(point1.y - point2.y) + abs(point1.z - point2.z)

                list_of_euclidean_distances.append(d)

    return list_of_euclidean_distances

def ottieni_features_delaunay(filename, num_frames):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    cap = cv2.VideoCapture(filename)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - num_frames - 1)

    ret, frame = cap.read()

    landmarks = []
    list_of_euclidean_distances = []
    lower_bound = 29

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

            if results and results.multi_face_landmarks:
                # Primo elemento della tupla
                point1 = results.multi_face_landmarks[0].landmark[i[0]]
                node1_x = int(point1.x * width)
                node1_y = int(point1.y * height)
                landmarks.append((node1_x,node1_y))

                # Secondo elemento della tupla
                point2 = results.multi_face_landmarks[0].landmark[i[1]]
                node2_x = int(point2.x * width)
                node2_y = int(point2.y * height)
                landmarks.append((node2_x,node2_y))

        landmarks_np = np.array(landmarks)
        landmarks_np = np.unique(landmarks_np, axis=0)
        print(np.shape(landmarks_np))
        landmarks_np = np.reshape(landmarks_np, (len(landmarks_np), 2))
        triangles = Delaunay(landmarks_np)

        # Set per tenere traccia dei segmenti unici
        unique_segments = set()

        # Itera sui triangoli
        for tri in triangles.simplices:
            pt1, pt2, pt3 = landmarks_np[tri]
            
            # Controllo per segmenti unici
            segment1 = tuple(sorted([tuple(pt1), tuple(pt2)]))
            segment2 = tuple(sorted([tuple(pt2), tuple(pt3)]))
            segment3 = tuple(sorted([tuple(pt3), tuple(pt1)]))
            
            # Disegna solo segmenti unici
            if segment1 not in unique_segments:
                unique_segments.add(segment1)
                dist = np.linalg.norm(pt2 - pt1)
                list_of_euclidean_distances.append(dist)
            if segment2 not in unique_segments:
                unique_segments.add(segment2)
                dist = np.linalg.norm(pt3 - pt2)
                list_of_euclidean_distances.append(dist)
            if segment3 not in unique_segments:
                unique_segments.add(segment3)
                dist = np.linalg.norm(pt3 - pt1)
                list_of_euclidean_distances.append(dist)

            if len(list_of_euclidean_distances) >= lower_bound:
                return list_of_euclidean_distances
        
        return list_of_euclidean_distances

def create_csv(csv_filename, directory, type, num_frames, experiment):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        header_string = []

        num_features = 0
        if experiment == "delaunay":
            num_features = 41
        else:
            num_features = 20
        
        for i in range(num_features):
            header_string.append("Feature " + str(i))

        header_string.append("Label")
        writer.writerow(header_string)

        if os.path.isdir(directory):
            files = glob.glob(directory + "/*.avi")

            for video in tqdm(files, desc=directory, ncols=100):
                res = []
                if experiment == "delaunay":
                    res = ottieni_features_delaunay(video, num_frames)
                else:
                    res = ottieni_features_da_video(video,type, num_frames)
                video_label = str(''.join(video.split("\\")[1].split(".")[0].split("_")[:4]))

                print("print shape of res: ",np.shape(res))

                if np.shape(res)[0] == num_features * num_frames:
                    res_split = np.array_split(res, num_frames)

                    for i in range(num_frames):
                        info_row = res_split[i]
                        writer.writerow(np.append(info_row, video_label))
                '''else:
                    print("Failed to fetch lip features from video: ",video)'''

if __name__ == "__main__":
    #create_data_correctly("train_dataset.csv", "test_dataset.csv")
    create_csv("delaunay_test.csv","Dataset/Test","",20,"delaunay")


