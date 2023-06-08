import pandas
import numpy as np
import os, glob, cv2
import math
import mediapipe as mp
import pandas
import psutil
import LipLandmarks as lp
import LipLandmarksDynamic as lpd
import LipLandmarksFullMesh as lpfm
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from tqdm import tqdm
from scipy.spatial import Delaunay
from collections import OrderedDict
import itertools
import matplotlib.pyplot as plt

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
        #'y_train': ohe_correct(y_train, num_distinct_people),
        #'y_test': ohe_correct(y_test, num_distinct_people),
        'y_train': ohe_correct(y_train, 256),
        'y_test': ohe_correct(y_test, 256),
        #Nel caso dovesse servire, fornisco anche la hashmap
        'hashmap': hashmap,
        'y_test_before_ohe': y_test,
        'y_train_before_ohe': y_train
    }

    return data

def reduce_data(data,num_columns):
    # Estrai l'ultima colonna dell'array

    dataset_train = np.column_stack((data['x_train'], data['y_train_before_ohe']))
    dataset_test = np.column_stack((data['x_test'], data['y_test_before_ohe']))
    dataset_completo = np.concatenate((dataset_train, dataset_test), axis=0)

    last_column = dataset_completo[:, -1]

    # Ottieni i distinti valori dell'ultima colonna
    distinct_values = np.unique(last_column)

    #print("Ci sono ",len(distinct_values))

    # Inizializza un dizionario per tenere traccia degli array di righe
    grouped_rows = {}

    # Itera sui distinti valori dell'ultima colonna
    for value in distinct_values:
        # Seleziona le righe che hanno il valore corrente nell'ultima colonna
        rows_with_value = dataset_completo[last_column == value]

        # Aggiungi le righe al dizionario
        grouped_rows[value] = rows_with_value

    list_of_variances = []
    
    for value in distinct_values:
        '''print("Value: ",value)
        print("Shape of grouped_rows[value]", np.shape(grouped_rows[value]))'''

        variance = np.var(grouped_rows[value][:, :-1], axis=0)
        list_of_variances.append(variance)
    
    #print("Shape of list of variances: ",np.shape(list_of_variances))

    variances_mean = np.mean(list_of_variances, axis=0)

    #print("Shape of variances_mean:",np.shape(variances_mean))
    #print("Variances Mean: ",variances_mean)

    indexes_of_lowest_variances_mean = np.argsort(variances_mean)[:num_columns]
    
    #print("Indexes of lowest variances mean:",indexes_of_lowest_variances_mean)

    reduced_data = {
        'x_train': data['x_train'][:,indexes_of_lowest_variances_mean],
        'x_test': data['x_test'][:,indexes_of_lowest_variances_mean],
        'y_train': data['y_train'],
        'y_test': data['y_test'],
        #Nel caso dovesse servire, fornisco anche la hashmap
        'hashmap': data['hashmap'],
        'y_test_before_ohe': data['y_test_before_ohe'],
        'y_train_before_ohe': data['y_train_before_ohe'],
    }

    return reduced_data

def print_stats(data):
    x_train_shape = np.shape(data['x_train'])
    x_test_shape = np.shape(data['x_test'])
    y_train_shape = np.shape(data['y_train'])
    y_test_shape = np.shape(data['y_test'])
    
    print("Shape of X_TRAIN:", x_train_shape)
    print("Shape of X_TEST:", x_test_shape)
    print("Shape of Y_TRAIN:", y_train_shape)
    print("Shape of Y_TEST:", y_test_shape)
    
def ottieni_features_twenty_spaced_da_video(filename, type, num_frames):
    # Apri il video
    video = cv2.VideoCapture(filename)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_interval = np.floor(total_frames/num_frames)

    # Verifica se il video è stato aperto correttamente
    if not video.isOpened():
        print("Impossibile aprire il video")
        return

    # Inizializza variabili
    frame_count = 0

    list_of_euclidean_distances = []

    while True:
        # Leggi il frame corrente
        ret, frame = video.read()

        # Verifica se il frame è stato letto correttamente
        if not ret:
            break

        # Mostra il frame solo se è un frame selezionato
        if frame_count % frame_interval == 0:
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


        frame_count += 1

    # Rilascia le risorse
    video.release()
    return list_of_euclidean_distances

def ottieni_features_twentytwo_spaced_da_video(filename, type, num_frames):
    # Apri il video
    video = cv2.VideoCapture(filename)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_interval = np.floor(total_frames/num_frames)

    # Verifica se il video è stato aperto correttamente
    if not video.isOpened():
        print("Impossibile aprire il video")
        return

    # Inizializza variabili
    frame_count = 0

    list_of_euclidean_distances = []

    while True:
        # Leggi il frame corrente
        ##printMemoryUsage("Sono al frame"+str(frame_count))
        ret, frame = video.read()
        ##printMemoryUsage("Sono al frame"+str(frame_count)+" dopo aver letto il frame")

        # Verifica se il frame è stato letto correttamente
        if not ret:
            break

        # Mostra il frame solo se è un frame selezionato
        if frame_count % frame_interval == 0:
            height, width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = face_mesh.process(rgb_frame)

            iterator = iter(lpd.lip_landmarks)
            for j in range(0, len(lpd.lip_landmarks)):
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


        frame_count += 1

    # Rilascia le risorse
    video.release()
    return list_of_euclidean_distances

def ottieni_features_twenty_da_video(filename,type, num_frames):
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

def ottieni_features_twentytwo_da_video(filename,type, num_frames):
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

        iterator = iter(lpd.lip_landmarks)
        for j in range(0, len(lpd.lip_landmarks)):
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

    list_of_euclidean_distances = []
    
    while ret:
        frame_distances = []
        landmarks = []

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

        if np.shape(landmarks)[0] == 0:
            return []
        else:

            landmarks_np = np.array(landmarks)

            #per frame diversi non trova gli stessi triangoli
            #l'algoritmo di delaunay individua a quali terne di punti associare i triangoli 
            #in base alla loro posizione euclidea
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
                    frame_distances.append(dist)
                if segment2 not in unique_segments:
                    unique_segments.add(segment2)
                    dist = np.linalg.norm(pt3 - pt2)
                    frame_distances.append(dist)
                if segment3 not in unique_segments:
                    unique_segments.add(segment3)
                    dist = np.linalg.norm(pt3 - pt1)
                    frame_distances.append(dist)
        
            for distance in frame_distances[:29]:
                list_of_euclidean_distances.append(distance)

    return list_of_euclidean_distances

def OLD_ottieni_features_fullmesh_spaced(filename,num_frames):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    cap = cv2.VideoCapture(filename)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_interval = np.floor(total_frames/num_frames)

    print("Frame interval:",frame_interval)

    ret, frame = cap.read()

    # Inizializza variabili
    frame_count = 0

    list_of_euclidean_distances = []
    
    while ret:
        landmarks = []

        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
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
                    cv2.circle(rgb_frame, (node1_x, node1_y), 1, (255, 255, 255), -1)

                    # Secondo elemento della tupla
                    point2 = results.multi_face_landmarks[0].landmark[i[1]]
                    node2_x = int(point2.x * width)
                    node2_y = int(point2.y * height)
                    landmarks.append((node2_x,node2_y))
                    cv2.circle(rgb_frame, (node1_x, node1_y), 1, (255, 255, 255), -1)

            
            # landmarks senza duplicati
            landmarks = list(OrderedDict.fromkeys(landmarks))
            print("Shape of landmarks:",np.shape(landmarks))
            if np.shape(landmarks)[0]!=18:
                plt.imshow(rgb_frame)
                plt.axis('off')  # Opzionale: per nascondere gli assi
                plt.show()

            # considero tutte le combinazioni di punti
            fullmesh = list(itertools.combinations(landmarks,2))

            last_shape = np.shape(list_of_euclidean_distances)
            # calcolo la distanza per ogni coppia
            for link in fullmesh:
                point1, point2 = link
                d = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1]- point1[1]) ** 2)
                list_of_euclidean_distances.append(d)

            print("For frame",frame_count,": the vector of distances is updated, inserted = ",np.shape(list_of_euclidean_distances)[0]-last_shape[0])
        
        frame_count += 1

    return list_of_euclidean_distances

def ottieni_features_fullmesh_spaced_da_video(filename, type, num_frames):
    # Apri il video
    video = cv2.VideoCapture(filename)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_interval = np.floor(total_frames/num_frames)

    # Verifica se il video è stato aperto correttamente
    if not video.isOpened():
        print("Impossibile aprire il video")
        return

    # Inizializza variabili
    frame_count = 0

    list_of_euclidean_distances = []

    while True:
        # Leggi il frame corrente
        ret, frame = video.read()

        # Verifica se il frame è stato letto correttamente
        if not ret:
            break

        # Mostra il frame solo se è un frame selezionato
        if frame_count % frame_interval == 0:
            height, width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = face_mesh.process(rgb_frame)
            #last_shape = np.shape(list_of_euclidean_distances)

            iterator = iter(lpfm.lip_landmarks)
            for j in range(0, len(lpfm.lip_landmarks)):
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
            
        #print(np.shape(list_of_euclidean_distances)[0]-last_shape[0],"segments were added at frame",frame_count)
        frame_count += 1
    #print("Final shape:",np.shape(list_of_euclidean_distances))
    # Rilascia le risorse
    video.release()
    return list_of_euclidean_distances

def ottieni_features_fullmesh(filename,num_frames):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    cap = cv2.VideoCapture(filename)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - num_frames - 1)

    ret, frame = cap.read()

    list_of_euclidean_distances = []
    
    while ret:
        landmarks = []

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

        # landmarks senza duplicati
        landmarks = list(OrderedDict.fromkeys(landmarks))

        # considero tutte le combinazioni di punti
        fullmesh = list(itertools.combinations(landmarks,2))

        # calcolo la distanza per ogni coppia
        for link in fullmesh:
            point1, point2 = link
            d = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1]- point1[1]) ** 2)
            list_of_euclidean_distances.append(d)


    return list_of_euclidean_distances           

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        header_string = []

        num_features = 0

        if experiment == "delaunay":
            num_features = 29
        elif "fullmesh" in experiment:
            num_features = 153
        elif "twentytwo" in experiment:
            num_features = 22
        else:
            num_features = 20

        if "behavioural" in experiment:
            num_features = num_features * num_frames

        print("Creando un dataset con ",num_features," features")
        
        for i in range(num_features):
            header_string.append("Feature " + str(i))
        
        if "behavioural" in experiment:
            num_features = num_features / num_frames
        
        header_string.append("Label")
        writer.writerow(header_string)
        file.close()
    

    if os.path.isdir(directory):
        files = glob.glob(directory + "/*.avi")
        res = []

        for video in tqdm(files, desc=directory, ncols=100):
            #print("Nome video: ", video)
            
            #twenty = 20 feature per frame, gli ultimi n_frame del video
            #twenty_dynamic = 22 feature per frame, gli ultimi n_frame del video

            #twenty_spaced = 20 feature per frame, n_frame a video presi a intervalli regolari
            #twenty_spaced_dynamic = 22 feature per frame, n_frame a video presi a intervalli regolari

            #twenty_spaced_behavioural = 20 feature per frame, n_frame a video presi a intervalli regolari, RIGHE CONCATENATE
            #twenty_spaced_dynamic_behavioural = 22 feature per frame, n_frame a video presi a intervalli regolari, RIGHE CONCATENATE

            #full_mesh = 153 feature per frame, gli ultimi n_frame del video

            #full_mesh_spaced = 153 feature per frame, n_frame a video presi a intervalli regolari
            #printMemoryUsage("Appena caricato il video")
            if experiment == "delaunay":
                res = ottieni_features_delaunay(video, num_frames)
            elif experiment == "twenty":
                res = ottieni_features_twenty_da_video(video, type, num_frames)
            elif experiment == "twentytwo":
                res = ottieni_features_twentytwo_da_video(video, type, num_frames)
            elif experiment == "twenty_spaced":
                res = ottieni_features_twenty_spaced_da_video(video,type,num_frames)
            elif experiment == "twentytwo_spaced":
                res = ottieni_features_twentytwo_spaced_da_video(video,type,num_frames)
            elif experiment == "twenty_spaced_behavioural":
                res = ottieni_features_twenty_spaced_da_video(video,type,num_frames)
            elif experiment == "twentytwo_spaced_behavioural":
                res = ottieni_features_twentytwo_spaced_da_video(video,type,num_frames)
            elif experiment == "fullmesh":
                res = ottieni_features_fullmesh(video,num_frames)
            elif experiment == "fullmesh_spaced":
                res = ottieni_features_fullmesh_spaced(video,type, num_frames)
            #printMemoryUsage("Appena misurato il video")

            video_label = str(''.join(video.split("\\")[1].split(".")[0].split("_")[:4]))

            print("\nShape of res: ",np.shape(res))
            if np.shape(res)[0] == num_features * num_frames:
                print("Per il video : ",video," sto inserendo: ",np.shape(res)[0])
                with open(csv_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    if "behavioural" in experiment:
                        writer.writerow(np.append(res,video_label))
                    else:
                        res_split = np.array_split(res, num_frames)
                        for i in range(num_frames):
                            info_row = res_split[i]
                            writer.writerow(np.append(info_row, video_label))
                    file.close()
            #printMemoryUsage("Appena scritto nel file")
            else:
                    print("Non è stato possibile prelevare le features dal video:",video)

def create_csv(csv_filename, directory, type, num_frames, experiment):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        header_string = []

        num_features = 0

        if experiment == "delaunay":
            num_features = 29
        elif "fullmesh" in experiment:
            num_features = 153
        elif "twentytwo" in experiment:
            num_features = 22
        else:
            num_features = 20

        if "behavioural" in experiment:
            num_features = num_features * num_frames

        #print("Creando un dataset con ",num_features," features")
        
        for i in range(num_features):
            header_string.append("Feature " + str(i))
        
        if "behavioural" in experiment:
            num_features = num_features / num_frames
        
        header_string.append("Label")
        writer.writerow(header_string)

        failed_videos = 0

        if os.path.isdir(directory):
            files = glob.glob(directory + "/*.avi")

            for video in tqdm(files, desc=directory, ncols=100):
                print("\nNome video: ", video)
                res = []

                #twenty = 20 feature per frame, gli ultimi n_frame del video
                #twenty_dynamic = 22 feature per frame, gli ultimi n_frame del video

                #twenty_spaced = 20 feature per frame, n_frame a video presi a intervalli regolari
                #twentytwo_spaced = 22 feature per frame, n_frame a video presi a intervalli regolari

                #full_mesh = 153 feature per frame, gli ultimi n_frame del video
                #full_mesh_spaced = 153 feature per frame, n_frame a video presi a intervalli regolari
                #full_spaced_behavioural = 153 features per frame, n frame a video presi a intervalli regolari, RIGHE CONCATENATE

                if experiment == "delaunay":
                    res = ottieni_features_delaunay(video, num_frames)
                elif experiment == "twenty":
                    res = ottieni_features_twenty_da_video(video, type, num_frames)
                elif experiment == "twentytwo":
                    res = ottieni_features_twentytwo_da_video(video, type, num_frames)
                elif experiment == "twenty_spaced":
                    res = ottieni_features_twenty_spaced_da_video(video,type,num_frames)
                elif experiment == "twentytwo_spaced":
                    res = ottieni_features_twentytwo_spaced_da_video(video,type,num_frames)
                elif experiment == "twenty_spaced_behavioural":
                    res = ottieni_features_twenty_spaced_da_video(video,type,num_frames)
                elif experiment == "twentytwo_spaced_behavioural":
                    res = ottieni_features_twentytwo_spaced_da_video(video,type,num_frames)
                elif experiment == "fullmesh":
                    res = ottieni_features_fullmesh(video,num_frames)
                elif experiment == "fullmesh_behavioural":
                    res = ottieni_features_fullmesh(video,num_frames)
                elif experiment == "fullmesh_spaced":
                    res = ottieni_features_fullmesh_spaced_da_video(video,type,num_frames)
                elif experiment == "fullmesh_spaced_behavioural":
                    res = ottieni_features_fullmesh_spaced_da_video(video,type,num_frames)

                video_label = str(''.join(video.split("\\")[1].split(".")[0].split("_")[:4]))

                print("Shape of res:",np.shape(res))

                if np.shape(res)[0] == num_features * num_frames:
                    print("Per il video : ",video," sto inserendo: ",np.shape(res)[0])
                    if "behavioural" in experiment:
                        writer.writerow(np.append(res,video_label))
                    else:
                        res_split = np.array_split(res, num_frames)
                        for i in range(num_frames):
                            info_row = res_split[i]
                            writer.writerow(np.append(info_row, video_label))
                else:
                    failed_videos = failed_videos + 1
                    print("Failed videos:",failed_videos)

def printMemoryUsage(message):
    pid = os.getpid()
    # Ottenere le informazioni sulla memoria del processo
    process = psutil.Process(pid)
    memory_info = process.memory_info()

    # Ottenere la quantità di memoria RAM utilizzata dal processo in bytes
    memory_usage = memory_info.rss

    # Convertire la quantità di memoria utilizzata in una forma leggibile
    formatted_memory_usage = psutil._common.bytes2human(memory_usage)

    print("Memoria (",message,")",formatted_memory_usage)

if __name__ == "__main__":
    print("Per usare read_data, chiama i suoi metodi da un tester")
    



    

    