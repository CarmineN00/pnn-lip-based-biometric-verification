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
from scipy.spatial.distance import cityblock

num_frames = 20

def ottieni_features_da_video(filename,type):
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
                if (type == "DistanzeEuclidee2D"):
                    cv2.circle(frame, (node1_x, node1_y), 2, (255, 255, 255), -1)

                # Secondo elemento della tupla
                point2 = results.multi_face_landmarks[0].landmark[i[1]]
                node2_x = int(point2.x * width)
                node2_y = int(point2.y * height)
                if (type == "DistanzeEuclidee2D"):
                    cv2.circle(frame, (node2_x, node2_y), 2, (255, 255, 255), -1)

                # Calcolo della distanza euclidea tra i punti
                if (type == "DistanzeEuclidee2D"):
                    d = math.sqrt((node2_x - node1_x) ** 2 + (node2_y - node1_y) ** 2)
                elif (type == "DistanzeEuclideeNormalizzate2D"):
                    d = math.sqrt((point2.x - point1.y) ** 2 + (point2.y - point1.y) ** 2)
                elif (type == "CityBlock3D"):
                    d = abs(point1.x - point2.x) + abs(point1.y - point2.y) + abs(point1.z - point2.z)

                list_of_euclidean_distances.append(d)

    return list_of_euclidean_distances

def create_csv(csv_filename, directory,type):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        header_string = []

        for i in range(20):
            header_string.append("Feature " + str(i))

        header_string.append("Label")
        writer.writerow(header_string)

        if os.path.isdir(directory):
            files = glob.glob(directory + "/*.avi")

            for video in tqdm(files, desc=directory+" using "+type, ncols=100):
                res = ottieni_features_da_video(video,type)
                video_label = str(''.join(video.split("\\")[1].split(".")[0].split("_")[:4]))

                if np.shape(res)[0] == 20 * num_frames:
                    res_split = np.array_split(res, num_frames)

                    for i in range(num_frames):
                        info_row = res_split[i]
                        writer.writerow(np.append(info_row, video_label))

if __name__ == "__main__":
    metriche = ["DistanzeEuclidee2D", "DistanzeEuclideeNormalizzate2D", "CityBlock3D"]

    for metrica in metriche:
        csv_filename = metrica + "_" + str(num_frames) + "_"
    
        test_csv_file = csv_filename + "test.csv"
        if not os.path.exists(test_csv_file):
            create_csv(test_csv_file, "Dataset/Test", metrica)
        else:
            print(test_csv_file, "già esiste")
    
        train_csv_file = csv_filename + "train.csv"
        if not os.path.exists(train_csv_file):
            create_csv(train_csv_file, "Dataset/Train", metrica)
        else:
            print(train_csv_file, "già esiste")

